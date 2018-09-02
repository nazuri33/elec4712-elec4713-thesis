clear
count = 200;

sequences = [1 2 3 4 5 6 7 8; 8 7 6 5 4 3 2 1; 1 9 9 9 9 9 9 8; 8 9 9 9 9 9 9 1];


for i = 1:count
    rand_index = randperm(4,1);
    rand_sequence = sequences(rand_index,:);
    input(i,:) = rand_sequence;
    
end



csvwrite('SORNOcclusionInput200.csv', input)