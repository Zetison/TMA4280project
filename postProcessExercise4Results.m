filename = 'results/exercise4.dat';
delimiterIn = ' ';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);

data = A.data(:,2:end);
P_arr = A.data(:,1);
k_arr = zeros(size(A.colheaders,2)-1,1);
for i = 1:length(A.colheaders)-1
    k_arr(i) = str2double(A.colheaders{i+1});
end

for i = 1:length(P_arr)
    fid = fopen(['results/exercise4_P' num2str(P_arr(i)) '.dat'],'wt+','b');
    
    fprintf(fid,'n time\n');
    for j = 1:length(k_arr)
        fprintf(fid, '%d %g\n', 2^k_arr(j), data(i,j));
    end
    fclose(fid);
end

for j = 1:length(k_arr)
    fid = fopen(['results/exercise4_S_p_k' num2str(k_arr(j)) '.dat'],'wt+','b');
    
    fprintf(fid,'P S_P\n');
    for i = 2:length(P_arr)
        fprintf(fid, '%d %g\n', P_arr(i), data(i,j)/data(1,j));
    end
    fclose(fid);
end

for j = 1:length(k_arr)
    fid = fopen(['results/exercise4_eta_p_k' num2str(k_arr(j)) '.dat'],'wt+','b');
    
    fprintf(fid,'P eta_P\n');
    for i = 2:length(P_arr)
        fprintf(fid, '%d %g\n', P_arr(i), data(i,j)/data(1,j)/P_arr(i));
    end
    fclose(fid);
end