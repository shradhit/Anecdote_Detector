# This script lists the files as python list
echo -n "["
A=($(ls))
for i in ${A[*]}
do
    if [ "$i" == "${A[-1]}" ]
    then
	echo -n "\"$i\"]"
    else
	echo -n "\"$i\", "
    fi
done
    
