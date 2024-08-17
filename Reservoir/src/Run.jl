using Reservoir, ReservoirComputing, Random, Plots

Random.seed!(42);		# fix seed to get repeatable results

alpha = 0.05;			# leaky coefficient for RNN
input_time = 300;
input_steps_per_unit = 20;
res_size = 20;			# set spectral radius to 1.0 and sparsity to 0.4
train_frac = 2/3;
input_complexity = 3;
offset_units = 2.0;
use_lasso = true;		# false => ridge regression or linear regression
lasso_loss = 7e-4;
ridge_loss = 1e-4;		# 0 => standard linear regression
show_train=false;

pl, test_output, train_output, output_layer = driver(input_time, input_steps_per_unit, 
		input_complexity, offset_units, res_size, train_frac, alpha, use_lasso, 
		lasso_loss, ridge_loss, show_train);
display(pl);

# output_layer.output_matrix[1:end]

savefig(pl, "~/Desktop/reservoir.pdf");

##############################################################################
## Notes

# test basic RNN calculation of states
# for i >= 1 and less than length(input[1,:])
test_state(7,input,esn)

# get field names
fieldnames(typeof(esn))

W=esn.reservoir_matrix;
Win=esn.input_matrix;
x=esn.states;

x[:,1]
tanh.(W*x[:,1]+Win*input[:,2])	# input time is offset, ith input drives ith state
x[:,2]		# 




