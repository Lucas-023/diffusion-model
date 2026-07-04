"""
app/backend/jobs.py
===================
Fila de jobs em memória com UM worker em thread.

Uma edição leva dezenas de segundos de GPU (50 passos DDIM x K forwards);
segurar o request HTTP aberto esse tempo todo é frágil. Em vez disso:
POST /api/edit enfileira e devolve job_id; o front faz polling em
GET /api/jobs/{id}. Um único worker garante que só um job usa a GPU por
vez — sem OOM por concorrência.
"""

import queue
import threading
import time
import traceback
import uuid


class Job:
    def __init__(self, params):
        self.id = uuid.uuid4().hex[:12]
        self.params = params
        self.status = "queued"      # queued | running | done | error
        self.stage = ""             # texto livre: "inversão", "edição", ...
        self.progress = 0.0         # 0..1 dentro do job
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "result": self.result,
            "error": self.error,
        }


class JobQueue:
    def __init__(self, worker_fn):
        self._worker_fn = worker_fn
        self._q = queue.Queue()
        self._jobs = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, job):
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job.id

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def pending(self):
        return self._q.qsize()

    def _loop(self):
        while True:
            job = self._q.get()
            job.status = "running"
            job.started_at = time.time()
            try:
                job.result = self._worker_fn(job)
                job.progress = 1.0
                job.status = "done"
            except Exception as e:
                traceback.print_exc()
                job.error = str(e)
                job.status = "error"
            finally:
                job.finished_at = time.time()
                self._q.task_done()
