mkdir -p $HOME/.local/lib

wget http://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb

dpkg -x libffi6_3.2.1-8_amd64.deb unpacked/

cp unpacked/usr/lib/x86_64-linux-gnu/libffi.so.6* $HOME/.local/lib/

rm -rf unpacked libffi6_3.2.1-8_amd64.deb
