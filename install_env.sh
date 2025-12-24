#!/bin/sh

work_dir=$(pwd)

install_python3_8(){
    echo "--Installation des dépendances--"
    sudo apt-get update
    sudo apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    openssl \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    libgdbm-dev


    echo "--Téléchargement de python 3.8--"
    cd /tmp
    wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
    tar xzf Python-3.8.12.tgz
    cd Python-3.8.12

    ./configure \
    --enable-optimizations \
    --with-openssl=/usr \
    --with-openssl-rpath=auto

    make -j$(nproc)
    sudo make altinstall


    echo "--Vérification--"
    python3.8 -V

    echo "--Nettoyage--"
    cd /opt
    sudo rm -f Python-3.8.12.tgz

    echo "=> PYTHON 3.8 FAIT !"
    cd "$work_dir"
    pwd
}

echo "Le projet necessite python3.8"
printf "Voulez vous l'installer [o/n]"
read -r usr_choice

case "$usr_choice" in 
  o|O|oui|Oui|y|Y|yes|YES)
    install_python3_8
    ;;
*)
    echo "Version de python3.8 trouvé : $(python3.8 -V)"
    ;;
esac

echo "--Création de l'env--"
python3.8 -m venv ./.env
. ./.env/bin/activate

echo "-- Mise à jour pip --"
pip install --upgrade pip setuptools wheel

echo "-- Installation du package ipykernel --"
pip install ipykernel