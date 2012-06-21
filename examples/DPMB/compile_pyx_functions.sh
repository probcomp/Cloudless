if [[ $(uname) != "Linux" ]] ; then 
    c:/Python27/Scripts/cython.py -a pyx_functions.pyx
    g++ -o pyx_functions.pyd -shared --verbose -fwrapv -O2 -Wall -fno-strict-aliasing -Ic:/Python27/include -Ic:/Python27/Lib/site-packages/numpy/core/include/ pyx_functions.c c:/Python27/python27.dll
else 
    cython -a pyx_functions.pyx
    gcc -o pyx_functions.so -shared -pthread -I/usr/include/python2.7 pyx_functions.c
fi
