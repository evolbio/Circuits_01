module Encoder
using Lux, Random, LossFunctions, Optimisers, Zygote, Plots, Printf,
			ConcreteStructs, Setfield, StatsBase
include("RNN_saf.jl")
include("Data.jl")
using .Data
include("Analysis.jl")
using .Analysis

####################################################################
# colors, based on standard Mathematica palette

mma = [RGB(0.3684,0.50678,0.7098),RGB(0.8807,0.61104,0.14204),
			RGB(0.56018,0.69157,0.19489), RGB(0.92253,0.38563,0.20918)];

####################################################################

export mma, fitness,calc_loss, calc_y_diff, calc_y_true, accuracy,
		 plot_data, state_dyn, gbm, ema, ema!, rw, log_dynamics,
		 plot_err_distn, vec_to_vec_matr, diff_autocor, sigmoidc,
		 RNN_saf, plot_log_all, plot_log_node1

vec_to_vec_matr(x::AbstractVector) = [reshape([i],1,1) for i in x]
matches(y_pred, y_true) = sum((y_pred .> 0.5) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
# log10 annualized fitness, log10 because y_true input on log10 scale
fitness(y_pred, y_diff) = 252*mean(map(x->ifelse(x>0.5,1,-1),y_pred) .* y_diff)
# diff between future and curr values
calc_y_diff(y, offset) = y[1+offset:end] - y[1:end-offset]
calc_y_true(y_diff) = map(x -> ifelse(x==0.0, Float64(rand(0:1)),
										ifelse(x>0.0,1.0,0.0)), y_diff)
ema_update(x, prev, alpha) = alpha*x + (1.0-alpha)*prev
# use sigmoidc to avoid 0 or 1 values, which throw error in crossentropy
sigmoidc(x; epsilon=1e-5) = clamp(1 / (1 + exp(-x)), epsilon, 1-epsilon)

function driver(T; rseed=0, adam=0.02, log2_nodes=3, rtol=1e-4, max_iter=100,
				offset=1, alpha=1, input=rw, ps=nothing, st=nothing)
	rng = Random.default_rng()
	if (rseed != 0) Random.seed!(rng, rseed) end
	s = log2_nodes
	nodes(s) = 2^s
	# output must use sigmoid activation for CrossEntropy loss
	if s > 1
		model = Chain(RNN_saf(1,nodes(s)), Dense(nodes(s),nodes(s-1),mish),
				Dense(nodes(s-1),nodes(s-2)), Dense(nodes(s-2),1,sigmoidc))
	else
		model = Chain(RNN_saf(1,nodes(s)), Dense(nodes(s),nodes(s-1)),
				Dense(nodes(s-1),1,sigmoidc))
	end
	if (ps === nothing) || (st === nothing)
		y,_ = input(T; alpha=alpha)
		ps,st = Lux.setup(rng, model)
		ym = @view vec_to_vec_matr(y)[1:end-offset]	
		yy = Encoder.cast_input_for_init(ym[1])	# Float16 matrix for initializing state
		_,st=model(yy,ps,st)					# initialize state
	end
	opt_state = create_optimiser(ps, adam)
	loss = y_pred = final_st = gs = 0
	loss_prev = 1e20
	loss_ema = match_ema = fitness_ema = 0.0
	for i in 1:max_iter
		try
			y,_ = input(T; alpha=alpha)
			y_diff = calc_y_diff(y, offset)
			y_true = calc_y_true(y_diff)
			# @printf("\nFreq positive changes = %5.3f\n\n", sum(y_true)/length(y_true))
			# input must be a matrix, so vector of matrices; chop offset off end
			ym = @view vec_to_vec_matr(y)[1:end-offset]	
			(loss, y_pred, final_st), back = 
						pullback(calc_loss, ym, model, ps, st, y_true)
			gs = back((one(loss), nothing, nothing))[3]
			opt_state, ps = Optimisers.update(opt_state, ps, gs)
			loss /= length(y_pred)
			loss_ema = ema_update(loss, loss_ema, 0.1)
			match_ema = ema_update(accuracy(y_pred,y_true), match_ema, 0.1)
			fitness_ema = ema_update(fitness(y_pred,y_diff), fitness_ema, 0.1)
			if i % 20 == 0
				@printf("Epoch %4d, ema: loss %9.4e, match %5.3f, fitness %6.3f\n", 
					i, loss_ema, match_ema, fitness_ema)
			end
			#if ((i == 50) && (case=="Brent")) Optimisers.adjust!(opt_state; eta=0.005) end
			if (loss < loss_prev && abs((loss_prev - loss)/loss) < rtol) break end
			loss_prev = loss
		catch e
			if isa(e, InterruptException)
				println("training loop terminated by interrupt")
				break
			else
				rethrow()
			end
		end
	end
	return loss, y, y_pred, model, ps, st, final_st, gs
end

# cost function for nodes=1
penalty(ps) = sum(abs,vcat(ps.layer_1.weight_ih, reduce(vcat,ps.layer_1.weight_hh), 
			reduce(vcat,ps.layer_2.weight), ps.layer_2.bias,
			reduce(vcat,ps.layer_3.weight),ps.layer_3.bias))

# Must use Zygote.Buffer to get around mutating array problem
function calc_loss(ym, model, ps, st, y_true)
	len_ym = length(ym)
	y_pred = Zygote.Buffer(zeros(len_ym))
	for i in 1:len_ym
		mout, st = model(ym[i], ps, st)
		y_pred[i] = mout[1,1]
	end
	yp = copy(y_pred)
	return sum(LossFunctions.CrossEntropyLoss(),yp,y_true), yp, st
end

function create_optimiser(ps, adam)
    opt = Optimisers.ADAM(adam)
    return Optimisers.setup(opt, ps)
end

function state_dyn(T, d)
	y,_ = d.input(T; alpha=d.alpha)
	ym = @view Encoder.vec_to_vec_matr(y)[1:end-d.os]
	len_ym = length(ym)
	y_pred = zeros(len_ym)
	st = Vector{typeof(d.s)}(undef,len_ym)
 	mout, st[1] = d.m(ym[1], d.p, d.s)
	y_pred[1] = mout[1,1]
	for i in 2:length(ym)
		mout, st[i] = d.m(ym[i], d.p, st[i-1])
		y_pred[i] = mout[1,1]
	end
	y[1:len_ym], st, y_pred
end

end # module Encoder
