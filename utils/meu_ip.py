import socket

def obter_ip_local():
    # Cria um socket UDP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Tenta conectar a um IP externo (Google DNS). 
        # Não envia dados de verdade nem precisa de internet ativa, 
        # serve apenas para o sistema revelar qual interface de rede usaria.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "Erro: Não foi possível descobrir o IP. Verifique a rede."
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    ip = obter_ip_local()
    print("\n" + "="*50)
    print(f"🌐 O Endereço IP deste computador é: {ip}")
    print("="*50 + "\n")
    print("Se este for o PC Mestre (Rank 0), use este IP no comando dos outros PCs:")
    print(f'--master_addr="{ip}"\n')