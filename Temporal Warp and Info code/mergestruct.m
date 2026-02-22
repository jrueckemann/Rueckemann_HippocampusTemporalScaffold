function [outstruct]=mergestruct(struct1,struct2)
%
%Jon Rueckemann 2023

%Obtain field names
fn1=fieldnames(struct1);
fn2=fieldnames(struct2);

%Obtain field values
val1=struct2cell(struct1);
val2=struct2cell(struct2);

%Merge
fn=[fn1(:); fn2(:)];
val=[val1; val2];
outstruct=reshape(cell2struct(val,fn),size(struct1));
end