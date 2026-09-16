ssh -N -R 17897:127.0.0.1:7890 root@hn01-ssh.gpuhome.cc -p 30557



export http_proxy=http://127.0.0.1:17897 https_proxy=http://127.0.0.1:17897 HTTP_PROXY=http://127.0.0.1:17897 HTTPS_PROXY=http://127.0.0.1:17897 ALL_PROXY=http://127.0.0.1:17897