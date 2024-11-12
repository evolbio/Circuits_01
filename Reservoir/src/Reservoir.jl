module Reservoir
using ReservoirComputing, Plots, MLJLinearModels
include("Illustration.jl")
using .Illustration
export make_esn, u, test_state, predict, plot_all, setup, driver,
	illustrate_driver, illustrate_driver_full, normalize!,
	print_mathematica

####################################################################
# colors, see MMAColors.jl in my private modules

mma = [RGB(0.3684,0.50678,0.7098),RGB(0.8807,0.61104,0.14204),
			RGB(0.56018,0.69157,0.19489), RGB(0.92253,0.38563,0.20918)];

####################################################################

function driver(input_time, input_steps_per_unit, input_complexity, 
		offset_units, res_size, train_frac, alpha, use_lasso, lasso_loss,
		ridge_loss, show_train)
	steps, input, offset = setup(input_time, input_steps_per_unit,
		input_complexity, offset_units);
	test_output, train_output, output_layer = predict(input, res_size,
		offset=offset, train_frac=train_frac, alpha=alpha,
		use_lasso=use_lasso, lasso_loss=lasso_loss, ridge_loss=ridge_loss);
	pl = plot_all(steps, input, input_steps_per_unit, offset, train_output,
		test_output, show_train=false)
	return pl, test_output, train_output, output_layer
end

# n is input complexity, number of sin terms used for input
ts(i,n) = i <= n ? 1 : 0
u(t,n) = sin(t)+ts(2,n)*sin(0.51*t)+ts(3,n)*sin(0.22*t)+
				ts(4,n)*sin(0.1002*t)+ts(5,n)*sin(0.05343*t)

function setup(input_time, input_steps_per_unit, input_complexity, offset_units)
	dt = 1/input_steps_per_unit
	steps = collect(1:dt:input_time);
	input_length = length(steps)
	input = reshape([u(t,input_complexity) for t in steps],1,input_length);
	offset=Int(offset_units*input_steps_per_unit);
	return steps, input, offset
end

# default spectral radius is 1.0, which means normalizing so that max abs
# eigenvalue is 1.0, keeps system from growing or shrinking
make_esn(;size=5, alpha=1.0, input=reshape([u(t,5) for t=1:100],1,100)) = 
		ESN(input;
    		reservoir = RandSparseReservoir(size, radius=1.0, sparsity=0.4),
    		reservoir_driver=RNN(leaky_coefficient=alpha),
    		input_layer = MinimumLayer())

# test basic RNN calculation of states
# tests if state[i+1] is approx equal to RNN calculation based on input and driver
# see https://docs.sciml.ai/ReservoirComputing/stable/esn_tutorials/lorenz_basic/
# for i >= 1 and less than length(input[1,:])
# x[t+1] = tanh.(W*x(t) + Win*u(t+1)), matches new input + old state => new state
test_state(i,input,esn) = isapprox(esn.states[:,i+1],
	tanh.(esn.reservoir_matrix*esn.states[:,i] + esn.input_matrix*input[:,i+1]))

# prediction, for input at time t, predict input at time t + offset
function predict(input, res_size; offset=2, train_frac=2/3,
					alpha=1.0, use_lasso=false, lasso_loss=3e-3, ridge_loss=1e-5)
	input_len = length(input[1,:])
	train_len = Int(round(train_frac * input_len))
	@assert offset + train_len <= input_len "offset + train_len > input_len"
	train_input = input[:, 1:train_len]
	train_target = input[:, offset+1:offset+train_len]
	esn = make_esn(size=res_size, input=train_input, alpha=alpha)
	# For linear, use ridge with small ridge_loss to prevent large coefficients
	# For L2 ridge loss, use ridge with larger ridge loss
	# Lasso (L1) with very small parameters converges to linear case but is
	# much slower. For L1 lasso case, adjust lasso gamma parameter (L1),
	# leave lambda (L2) parameter at 0. Issue is that Ridge can use fast
	# solver whereas ElasticNet (Lasso) must use slow solver. Small tol needed.
	lasso = LinearModel(;regression=ElasticNetRegression, 
			    solver=ProxGrad(max_iter=2000000, tol=1e-9),
				regression_kwargs=(lambda=0, gamma=lasso_loss))
	solver = use_lasso ? lasso : StandardRidge(ridge_loss)
	output_layer = train(esn, train_target, solver)
	train_output = esn(Predictive(train_input), output_layer)
	
	# Finish here and test offsets in above
	test_input = input[:,train_len+1:end]
	test_output = esn(Predictive(test_input), output_layer)
	
	return test_output, train_output, output_layer
end

function plot_all(steps, input, input_steps_per_unit, 
						offset, train_output, test_output; show_train=false)
	train_len = length(train_output[1,:])
	test_len = length(test_output[1,:])

	incr = show_train ? 1 : 0
	default(lw=1.5)
	plot(layout=(3+incr,1),legend=false, grid=false, palette=palette(mma))

	if show_train
		plot!(steps[1+2*offset:train_len],input[1,1+2*offset:train_len])
		plot!(steps[1+2*offset:train_len],train_output[1,1+offset:end-offset])
	end

	plot!(steps[train_len+1+2*offset:end],
				input[1,train_len+1+2*offset:end],subplot=1+incr,
				annotation=((0.035,0.96),("(a)",11)))
	plot!(steps[train_len+1+offset:end],
							test_output[1,1:end-offset],subplot=1+incr)

	# fix this if changing total time
	start = 235*input_steps_per_unit
	stop = 265*input_steps_per_unit
	plot!(steps[start:stop],input[1,start:stop],subplot=2+incr,
		ylabel="Environmental signal", annotation=((0.035,0.99),("(b)",11)))
	plot!(steps[start:stop],
		test_output[start-offset-train_len:stop-offset-train_len],subplot=2+incr)

	plot!(steps[start:stop],input[1,start:stop],subplot=3+incr, xlabel="Time")
	plot!(steps[start:stop],test_output[start-train_len:stop-train_len],
		subplot=3+incr, annotation=((0.035,0.99),("(c)",11)))
end

end # module Reservoir
