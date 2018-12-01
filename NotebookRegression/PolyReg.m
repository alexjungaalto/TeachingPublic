clear all; 
close all; 

%% read in the raw data from csv file
X = csvread('BTCETHClosing.csv') ; 
[N d] = size(X); 

%% randomly choose N_train data points as training set
N_train=10; 
indices = randperm(N);   % randomly permute the data point indices 1,..,N
X_train = X(indices(1:N_train),:) ; 

%% random choose N_val data points as validation set
N_val = 11;  
indices = randperm(N);  % randomly permute the data point indices 1,..,N
X_val = X(indices(1:N_val),:) ; 


% extract labels and features of training data 
x=X_train(:,1) ; % features of data points in trainig set
x=x/max(x);      % normalize features

y=X_train(:,2) ; % label of data points in training set
y=y/max(y);      % normalize labels

% extract labels and features of validation data 

x_val=X_val(:,1) ; % features of data points in validation set
x_val=x_val/max(x_val);    % normalize features

y_val=X_val(:,2) ; % label of data points in validation set
y_val=y_val/max(y_val); % normalize labels

%%%%%%%%
% polynomial regression with squared error loss
%%%%%%%%%%
max_degree = 4;
d = max_degree+1 ; % number of coefficients for x^0,x^1,...x^{max_degree}


%%% construct the feature matrix X_poly over training set for polynomial regression
X_poly =zeros(N_train,d)  ;
for iter_degree=1:d
    X_poly (:,iter_degree) = x.^(iter_degree-1) ; % add new feature x^(iter_degree-1)
end
X_poly(:,1)=ones(N_train,1); % set first feature to x^0 = 1 

%%% construct feature matrix for many possible feature values 
x_grid = linspace(min(x),max(x),1000); 
X_poly_grid =ones(1000,d)  ;
for iter_degree=1:d
    X_poly_grid (:,iter_degree) = x_grid.^(iter_degree-1) ; % add new feature x^(iter_degree-1)
end
X_poly_grid(:,1)=ones(1000,1);  % set first feature to x^0 = 1


%% compute optimal weight vector using closed-form
w_poly =  pinv(X_poly'*(X_poly) )*(X_poly')*y ; 
%% compute predicted labels using weight vector 
y_hat_poly = X_poly_grid * w_poly;

%% now use gradient descent to compute (approx.) the optimal weight vector 
w_GD = zeros(d,1);
NUM_ITER=1000000; 
logobj=zeros(NUM_ITER,1);

for iter_GD=1:NUM_ITER
    tmp = w_GD + (1/(2*N_train)) *(X_poly')* (y- X_poly*w_GD); 
    logobj(iter_GD) = norm(y- X_poly*w_GD); 
    w_GD=tmp;
end

%% cmopute predicted labels using weight vector produced by gradient descent
y_hat_GD =  X_poly_grid * w_GD;


%% scatter plot of training data
training=scatter(x,y,100,'LineWidth',3); 
hold on; 
%% add scatter plot of validation data
validation = scatter(x_val,y_val,100,'x','LineWidth',3) ; 


%% add predictions obtained from close-form solution for poly. reg.
closeform=plot(x_grid,y_hat_poly,'LineWidth',2); 
%% add predictinos obtained from poly.reg. with gradient descent
GDpred=plot(x_grid,y_hat_GD,'LineWidth',2); 

%% add labels of axis and the legend
xlabel("Normalized Bitcoin closing price",'FontSize',14); 
ylabel("Normalized Ethereum closing price",'FontSize',14); 
legend([training,validation,closeform,GDpred], ["training data","validation data","polyreg closed form ", "polyreg  GD"],'FontSize',20);