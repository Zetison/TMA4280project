filename = 'results/exercise2.dat';
delimiterIn = ' ';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);

data = A.data(:,2:end);
P_arr = A.data(:,1);
t_arr = zeros(size(A.colheaders,2)-1,1);
for i = 1:length(A.colheaders)-1
    t_arr(i) = str2double(A.colheaders{i+1});
end

for i = 1:length(P_arr)
    fid = fopen(['results/exercise2_k12_P' num2str(P_arr(i)) '.dat'],'wt+','b');
    
    fprintf(fid,'t time\n');
    for j = 1:length(t_arr)
        fprintf(fid, '%d %g\n', t_arr(j), data(i,j));
    end
    fclose(fid);
end

for j = 1:length(t_arr)
    fid = fopen(['results/exercise2_k12_t' num2str(t_arr(j)) '.dat'],'wt+','b');
    
    fprintf(fid,'P time\n');
    for i = 1:length(P_arr)
        fprintf(fid, '%d %g\n', P_arr(i), data(i,j));
    end
    fclose(fid);
end