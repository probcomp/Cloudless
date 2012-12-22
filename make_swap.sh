swapfile=$1
if [[ -z $swapfile ]] ; then
    swapfile=/mnt/swapfile
fi

if [[ ! -f $swapfile ]] ; then
    dd if=/dev/zero of=$swapfile bs=1G count=2
    mkswap $swapfile
    swapon $swapfile
fi
