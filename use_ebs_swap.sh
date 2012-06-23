if [[ `mount | grep xvda3` ]] ; then
    sudo swapoff /dev/xvda3
    sudo umount /dev/xvdm
    sudo mkswap /dev/xvdz
    sudo swapon /dev/xvdz
fi
