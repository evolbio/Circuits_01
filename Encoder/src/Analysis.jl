module Analysis
using Printf, StatsBase, Plots, Measures, Encoder, Lux, Random
export plot_err_distn, diff_autocor, log_dynamics, plot_data, plot_log_all,
		plot_log_node1

rng_state(rstate) = 
	rstate === nothing ?
		println(copy(Random.default_rng())) :
		copy!(Random.default_rng(), rstate)

# show how often prior diff predicts next diff
function prior_next_match(T,d; yy=nothing, prnt=true)
	if yy === nothing yy = d.input(T; alpha=d.alpha)[1] end
	tt=Vector{Int}(undef,length(yy)-2);
	for i in eachindex(tt)
		tt[i] = ifelse((yy[i+1] > yy[i]) == (yy[i+2] > yy[i+1]), 1, 0)
	end
	mtch = sum(tt)/length(tt)
	if prnt @printf("prior-next match = %5.3f\n", mtch) end
	return mtch
end

# plot match freq distn
function plot_match_distn(T,d; n=100)
	mtch = zeros(n);
	for i in 1:n
		_, _, yp, yd = pred_val(T,d);
		mtch[i] = accuracy(yp,calc_y_true(yd));
	end
	display(histogram(mtch,legend=:none))
	prior_next_match(5000,d)
	@printf("mean = %5.3f, sd = %5.3f\n", mean(mtch), std(mtch))
end

# plot match freq distn
function plot_err_distn(T,d; n=100, skip=0, rstate = nothing, match_test=2000)
	@assert 0.02*n - floor(0.02*n) == 0
	ytk = Int(0.02*n)
	rstate === nothing ?
		println(copy(Random.default_rng())) :
		copy!(Random.default_rng(), rstate)
	err = zeros(n);
	for i in 1:n
		_, yy, yp, yd = pred_val(T,d);
		yy = @view yy[1+skip:end]
		yp = @view yp[1+skip:end]
		yd = @view yd[1+skip:end]
		err[i] = prior_next_match(T,d; yy=yy, prnt=false) - accuracy(yp,calc_y_true(yd))
	end
	pl = histogram(err,legend=:none, color=mma[1],
		xlabel="Deviation from maximum accuracy", ylabel="Frequency",
		yticks=(ytk:ytk:4*ytk,string.(0.02:0.02:0.08)))
	max_acc = prior_next_match(match_test,d)
	annotate!((0.9,0.85),text(@sprintf("%11s = %5.3f", "max accuracy", max_acc),11, :right))
	annotate!((0.9,0.76),text(@sprintf("%11s = %5.3f", "median", median(err)),11, :right))
	annotate!((0.9,0.67),text(@sprintf("%11s = %5.3f", "mean", mean(err)),11, :right))
	annotate!((0.9,0.58),text(@sprintf("%11s = %5.3f", "standard dev", std(err)),11, :right))
	display(pl)
	@printf("median = %7.2e, mean = %7.2e, sd = %7.2e\n", median(err), mean(err), std(err))
	return pl
end

# predictions above and below mean reversion point (gbm w/mean reversion)
function pred_above_below(T,d)
	ts, yy, yyp, yyd = pred_val(T,d);
	yy = yy[1:end-d.os];
	mdiff = 0.8;
	cut = 5.166;
	cut = 0;
	cuta = cut + mdiff;
	cutb = cut - mdiff;
	i_above = findall(x->x>cuta,yy);
	i_below = findall(x->x<cutb,yy);
	mean(yyp[i_above])
	mean(yyp[i_below])
	corkendall(yy,yyp)
	autocor(yyd,[1,2])
end

function diff_autocor(T,d)
	_, _, _, yyd = pred_val(T,d);
	autocor(yyd,[1,2])
end

function plot_data(T,d; rstate = nothing)
	rstate === nothing ?
		println(copy(Random.default_rng())) :
		copy!(Random.default_rng(), rstate)
	pl=plot(layout=(4,2), size=(900,200*4), legend=:none)
	for i in 0:1
		ts, y_val, y_pred, y_diff = pred_val(T, d);
		y_fit = map(x-> x > 0.5 ? 1 : -1 , y_pred) .* y_diff;
		pushfirst!(y_fit,0.0);
		y_fit_sum = cumsum(y_fit);	# need to fix this if offset > 1
		
		@printf("\nDays per switch = %.2f\n", length(y_pred)/switches(y_pred));
		@printf("match = %.2f\n", accuracy(y_pred,calc_y_true(y_diff)));
		
		warmup = 0;	# skip first 100 obs to warmup model
		w1 = 1+warmup:length(ts);
		w2 = 1+warmup:length(y_fit_sum);
		mw1 = mid_series(w1);
		mw2 = mid_series(w2);

		lm = i == 0 ? 1cm : 0cm
		plot!(ts[w1],y_val[w1],subplot=1+i, color=mma[1], w=2,left_margin=lm)
		plot!(ts[w2],y_fit_sum[w2],subplot=3+i, color=mma[1], w=2,left_margin=lm)
		plot!(ts[mw1],y_val[mw1],subplot=5+i, color=mma[1], w=2,left_margin=lm)
		plot!(ts[mw2],y_fit_sum[mw2],subplot=7+i, color=mma[1], w=2,
			bottom_margin=1cm,left_margin=lm)
	end
	annotate!(pl[7],(1.08,-0.32),"Temporal sample points",16)
	annotate!(pl[1],(-0.16,0.5),text("Input value",12,rotation=90))
	annotate!(pl[5],(-0.16,0.5),text("Input value",12,rotation=90))
	annotate!(pl[3],(-0.16,0.5),text("Cumulative fitness",12,rotation=90))
	annotate!(pl[7],(-0.16,0.5),text("Cumulative fitness",12,rotation=90))
	chrs = 'a':'z'
	subp = vcat(1:2:7,2:2:8)
	for i in 1:8
		annotate!(pl[subp[i]],(0.06,0.97),text(@sprintf("(%s)",chrs[i]),10))
	end
	display(pl)
	return pl
end

# for a series with indices n1:n2, get the indices for the middle frac
function mid_series(s; frac=0.05)
	ls = length(s)
	frac /= 2
	mid = div(ls,2)
	mid - Int(round(frac*ls)):mid+Int(round(frac*ls))
end

function switches(y_pred)
	ydir = map(x -> x > 0.5 ? 1 : 0,y_pred);
	sw = 0
	for i in 2:lastindex(ydir)
		if (ydir[i] != ydir[i-1]) sw +=1 end
	end
	return sw
end

# get the predicted values
function pred_val(T, d)
	y,_ = d.input(T; alpha=d.alpha)
	ts = collect(1:length(y))
	ym = @view Encoder.vec_to_vec_matr(y)[1:end-d.os]
	y_diff = calc_y_diff(y, d.os)
	y_true = calc_y_true(y_diff)
	_, y_pred, _ = calc_loss(ym, d.m, d.p, d.s, y_true)
	return ts, y, y_pred, y_diff
end

struct PrintLayer <: Lux.AbstractExplicitLayer end

@inline function (prn::PrintLayer)(x, ps, st::NamedTuple)
	println("x = ", x)
	x, st
end

## state[t] used to calculate state[t+1] == layer[2][t]
## in which layer[2][t] is the output of the RNN layer[1]
function log_dynamics(T,d; logfile="/Users/steve/Desktop/output/model.log",
			rstate = nothing)
	rng_state(rstate)
	s = d.nodes
	nodes(s) = 2^s
	if s > 1
		model = Chain(
					PrintLayer(), RNN_saf(1,nodes(s)),
					PrintLayer(), Dense(nodes(s),nodes(s-1),mish),
					PrintLayer(), Dense(nodes(s-1),nodes(s-2)),
					PrintLayer(), Dense(nodes(s-2),1,sigmoidc))
	else
		model = Chain(
					PrintLayer(), RNN_saf(1,nodes(s)),
					PrintLayer(),Dense(nodes(s),nodes(s-1)),
					PrintLayer(),Dense(nodes(s-1),1,sigmoidc))
	end
	
	y,_ = d.input(T; alpha=d.alpha)
	ym = @view Encoder.vec_to_vec_matr(y)[1:end-d.os]
	ps,st = Lux.setup(Random.default_rng(), model)
	yy = Encoder.cast_input_for_init(ym[1]);	# Float16 matrix for initializing state
	_,st=model(yy,ps,st)						# initialize state
	
	# inject fitted parameters into new parameter tuple
	for i in 2:2:length(ps)
		dic = Dict(keys(ps)[i] => d.p[i÷2])
		ps = merge(ps, (; dic...))
	end

	io = open(logfile, "w")
	original_stdout = stdout # Save the original stdout
	redirect_stdout(io)
	println("st = ", Float64.(st.layer_2.hidden_state))
	
	for i in eachindex(ym)
		mout, st = model(ym[i], ps, st)
		yp = mout[1,1]
		i == lastindex(ym) ?
			println("yp = [", yp, ";;]") :
			println("yp = [", yp, ";;]", "\nst = ", st.layer_2.hidden_state)
	end
	redirect_stdout(original_stdout)
	close(io)
	return parse_log(length(ym), d.nodes, d.nodes>1 ? 3 : 2, logfile)
end

function parse_lines(line::String)
    # Check if the line starts with "st = ", "x = ", or "yp = " and extract values
    if occursin(r"^st = \[(.*?);;\]", line)
        return :st, parse.(Float64, split(match(r"^st = \[(.*?);;\]", line).captures[1], ";"))
    elseif occursin(r"^x = \[(.*?);;\]", line)
        return :x, parse.(Float64, split(match(r"^x = \[(.*?);;\]", line).captures[1], ";"))
    elseif occursin(r"^yp = \[(.*?);;\]", line)
        return :yp, parse.(Float64, split(match(r"^yp = \[(.*?);;\]", line).captures[1], ";"))
    end
end

function parse_log(len_input, log2_nodes, layers, logfile)
	n = 2^log2_nodes
    st_values = zeros(len_input,n)				# RNN layer state values
	layer_input = [zeros(len_input,2^i) for i 	# layer 1 = model input
					in pushfirst!(reverse(collect(0:log2_nodes)),0)]
    output = ones(len_input)					# model output
	i = j = 1
    open(logfile, "r") do file
        for line in eachline(file)
        	data_type, out = parse_lines(line)
           	if data_type == :st
            	st_values[i,:] = out
            elseif data_type == :x
            	layer_input[j][i,:] = out
                j += 1
            elseif data_type == :yp
            	output[i] = out[1]
                i += 1
                j = 1
            end
        end
    end
    if i != len_input+1 println("Error parsing log file\n") end
    return st_values, layer_input, output
end

function plot_log_all(d, st, layer, y_pred)
	strt = 30
	y = @view layer[1][:,1]
	@printf("\nDays per switch = %.2f\n", length(y_pred)/switches(y_pred));
	@printf("match = %.2f\n", accuracy(y_pred[1:end-d.os],calc_y_true(calc_y_diff(y,d.os))));
	pl = plot(layout=(length(layer),1), legend=:none)
	plot!([st[strt:end,:],y[strt:end]],subplot=1)
	for i in 2:lastindex(layer)
		plot!([layer[i][strt:end,:]],subplot=i)
	end
	display(pl)
	return pl
end

# state 1 output gives level of current input +/- current slope because
# slope provides significant information about sign of next change
# subtracting off prior input in state 2 then gives sign of most recent
# change plus momentum from most recent change, provides prediction of next
# change by distance from zero, which is needed for input into sigmoid function
# Underlying input is an ema of a random walk. Thus, to predict sign of next
# change, need to predict ema(t+1) - ema(t) = alpha*dW + (1-alpha)(ema(t)-ema(t-1)),
# where dW is N(0,1) RV, which is delta X_t of underlying process used to make ema
# state[t] used to calculate state[t+1] == layer[2][t]
# output 2 of RNN layer reflects prior input value

function plot_log_node1(d, st, layer, y_pred; r=143:257)
	y = layer[1][:,1]		# inputs
	st1 = layer[2][:,1]		# state 1 output from first layer, input into 2nd layer
	st2 = layer[2][:,2]	# state 2 output from first layer, input into 2nd layer
	@printf("\nDays per switch = %.2f\n", length(y_pred)/switches(y_pred))
	@printf("match = %.2f\n", accuracy(y_pred[1:end-d.os],calc_y_true(calc_y_diff(y,d.os))))
		
	# show match for shifted input, state 2 output is prior input value * constant value
	pl = plot(layout=(3,1), size=(500,600), legend=:none, left_margin=0.8cm)
	plot!(r,[y[r.-1],st2[r]*mean(y[r.-1])/mean(st2[r])],subplot=1,w=1.5,
		color=[mma[1] mma[2]],yticks=0.55:0.05:0.65)
	plot!(r,[y[r],st1[r]*mean(y[r])/mean(st1[r])],subplot=2,w=1.5,color=[mma[1] mma[2]])
	plot!(r,2*(y_pred[r].-0.5),subplot=3,w=1.5,color=[mma[1] mma[2]],bottom_margin=0.6cm)
	annotate!(pl[3],(0.5,-0.25),"Temporal sample points",12)
	annotate!(pl[1],(-0.13,0.5),text("Input value",11,rotation=90))
	annotate!(pl[2],(-0.13,0.5),text("Input value",11,rotation=90))
	annotate!(pl[3],(-0.13,0.5),text("Predicted direction",11,rotation=90))
	chrs = 'a':'z'
	for i in 1:3
		annotate!(pl[i],(0.05,0.99),text(@sprintf("(%s)",chrs[i]),9))
	end
	display(pl)
	return pl
end

# get distn of run lengths in sequence of 0,1 values in y_true
# shows how often direction of change continues in same direction
function count_runs(bits::Vector{Int})
    # Initialize the counter for the current run length and the vector to hold run counts.
    # Using a Dict for flexibility with run lengths.
    current_run_length = 1
    run_counts = Dict{Int, Int}()
    
    # Iterate through the bits starting from the second one
    for i in 2:length(bits)
        if bits[i] == bits[i-1]
            # If the current bit is the same as the previous, increment the run length
            current_run_length += 1
        else
            # If the run ended, increment the count for this run length in the dictionary
            run_counts[current_run_length] = get(run_counts, current_run_length, 0) + 1
            current_run_length = 1 # Reset for the next run
        end
    end
    
    # Account for the last run
    run_counts[current_run_length] = get(run_counts, current_run_length, 0) + 1
    
    # Convert the dictionary to a vector, filling in zeros where necessary
    max_run_length = maximum(keys(run_counts))
    run_length_vector = zeros(Int, max_run_length)
    for (length, count) in run_counts
        run_length_vector[length] = count
    end
    
    return run_length_vector
end

end # module Analysis