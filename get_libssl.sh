wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb

dpkg -x libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb unpacked/

cp unpacked/usr/lib/x86_64-linux-gnu/libssl.so.1.1 $HOME/.local/lib/
cp unpacked/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 $HOME/.local/lib/

rm -rf unpacked libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb
