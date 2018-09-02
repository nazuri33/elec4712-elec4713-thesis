clear
count = 200

for i = 1:count
    A(i,i) = 1
    A(i+count,:) = 0
end

k = 0

for j = 2*count+1:3*count
    A(j, count-k) = 1
    k = k+1
    A(j+count,:) = 0
end

csvwrite('200activationarray.csv', A)