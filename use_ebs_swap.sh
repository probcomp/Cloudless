if [[ ! `swapon -s | grep xvdz` ]] ; then
    sudo umount /dev/xvdm
    sudo mkswap /dev/xvdz
    sudo swapon /dev/xvdz
fi
