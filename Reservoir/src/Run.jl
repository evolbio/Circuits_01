using Reservoir, ReservoirComputing, Random, Plots

##############################################################################
## For plot in main text, Figure 2

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
## Plot for illustrative example of how reservoir computing works, Figure 1

# Send input data and esn.states to Mathematica to make plots
# Ignore plotting routines in Illustration.jl, not very good

# for top row of plots, match each input to each esn.state
input_up = reshape(0.1:0.2:0.9,1,5);
res_size=5;
weight=0.5;
esn=illustrate_driver(input_up, res_size; weight=weight);
esn.states

# for second row of plots, match each input to each esn.state
input_flat = 0.7*ones(1,5);
res_size=5;
weight=0.5;
esn=illustrate_driver(input_flat, res_size; weight=weight);
esn.states

# for third row of plots, match each input to each esn.state
input_down = reshape(reverse(0.1:0.2:0.9),1,5);
res_size=5;
weight=0.5;
esn=illustrate_driver(input_down, res_size; weight=weight);
esn.states

##############################
# for bottom row of plots
res_size = 10;
sparsity = 0.5;

Random.seed!(42);
input_up = reshape(0.0:0.1:1.0,1,11);
esn=ESN(input_up; reservoir=RandSparseReservoir(res_size; radius=1.0, 
	sparsity=sparsity));
y1=esn.states[:,end];

Random.seed!(42);
input_flat = 0.7*ones(1,11);
esn=ESN(input_flat; reservoir=RandSparseReservoir(res_size; radius=1.0, 
	sparsity=sparsity));
y2=esn.states[:,end];

Random.seed!(42);
input_down = reshape(reverse(0.0:0.1:1.0),1,11);;
esn=ESN(input_down; reservoir=RandSparseReservoir(res_size; radius=1.0, 
	sparsity=sparsity));
y3=esn.states[:,end];

Random.seed!(42);
input_sin = reshape([0.5*(sin(2π*t) + 1) for t in 0.0:0.1:1.0],1,11);;
esn=ESN(input_sin; reservoir=RandSparseReservoir(res_size; radius=1.0,
	sparsity=sparsity));
y4=esn.states[:,end];

states = hcat(y1,y2,y3,y4)
normalize!(states)
print_mathematica(states)	# output can be pasted into Mathematica

# The final graphic was made in Wolfram Mathematica. The code is included
# in the file box_figs.m. Figure 1 in manuscript

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




