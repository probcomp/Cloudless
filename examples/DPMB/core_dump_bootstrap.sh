#!/bin/sh

chk_root () {
    if [ ! $( id -u ) -eq 0 ]; then
        exec sudo sh ${0}
        exit ${?}
    fi
}

chk_root

mkdir /tmp/cores
chmod -R 1777 /tmp/cores
echo "\n* soft core unlimited" >> /etc/security/limits.conf
echo "ulimit -c unlimited" >> /etc/profile
echo "/tmp/cores/core.%e.%p.%h.%t" > /proc/sys/kernel/core_pattern
