# -*- coding: utf-8 -*-
"""AutoDL 远程助手（paramiko——密码认证——Windows 无 sshpass 的替代）
用法：
  python autodl_remote.py run "cmd"              # 执行命令（流式输出）
  python autodl_remote.py runbg "cmd"            # nohup 后台执行
  python autodl_remote.py put <local> <remote>
  python autodl_remote.py get <remote> <local>
  python autodl_remote.py check                  # 连接+GPU+系统探测
连接信息在文件底部常量（AutoDL 临时实例——会话内有效）
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import paramiko

HOST, PORT, USER, PWD = 'connect.westb.seetacloud.com', 53005, 'root', '9djpV2AqIlcp'


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return c


def run(cmd, timeout=3600, quiet=False):
    c = connect()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=True)
    out = ''
    for line in iter(stdout.readline, ''):
        out += line
        if not quiet:
            sys.stdout.write(line)
            sys.stdout.flush()
    code = stdout.channel.recv_exit_status()
    c.close()
    return code, out


def runbg(cmd):
    c = connect()
    full = f'nohup bash -lc {paramiko.util._quote(cmd)} > /root/autodl-tmp/bg.log 2>&1 & echo $!'
    stdin, stdout, stderr = c.exec_command(full, timeout=30)
    pid = stdout.read().decode().strip()
    c.close()
    print(f'后台 PID: {pid}')
    return pid


def put(local, remote):
    c = connect()
    s = c.open_sftp()
    s.put(local, remote)
    s.close()
    c.close()
    print(f'上传 {local} -> {remote}')


def get(remote, local):
    c = connect()
    s = c.open_sftp()
    s.get(remote, local)
    s.close()
    c.close()
    print(f'下载 {remote} -> {local}')


def check():
    code, out = run('nvidia-smi; echo ---; python3 --version; echo ---; df -h /root/autodl-tmp | tail -1; echo ---; ls /root/autodl-tmp', timeout=60)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        run(' '.join(sys.argv[2:]))
    elif cmd == 'runbg':
        runbg(' '.join(sys.argv[2:]))
    elif cmd == 'put':
        put(sys.argv[2], sys.argv[3])
    elif cmd == 'get':
        get(sys.argv[2], sys.argv[3])
    elif cmd == 'check':
        check()
