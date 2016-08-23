#!/usr/bin/bash
echo "Cleaning : rm *~ *.pyc __pycache__ *.so"
find . \( -name "*.~" -o -name "*.pyc" -o -name "*.so" \) -exec ls -l {} \;

PS3='> '   # le prompt
LISTE=("[y] yes" "[n]  no")  # liste de choix disponibles
select CHOIX in "${LISTE[@]}" ; do
    case $REPLY in
        1|y)
        find . \( -name "*.~" -o -name "*.pyc" -o -name "*.so" \) -exec rm -r {} \;
        break
        ;;
        2|n)
        echo ko
        break
        ;;
    esac
done

cd kalman/libs
sh ../../README
ln -s fortran_libs_py3.cpython-35m-x86_64-linux-gnu.so fortran_libs_py3.so
cd ../..

