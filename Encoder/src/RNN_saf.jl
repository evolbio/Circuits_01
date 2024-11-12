# Custom RNN layer
using NNlib

##############################################################
# Works only if RNN_saf layers are the first ones in the line,
#   and no RNN_saf layers after some other layer type
##############################################################

# Problem: first call with input x must initialize hidden state.
# Subsequent calls must do calculations. To achieve this, change
# x to Float16 matrix, make first model call with new x, then run
# model normally after that with original x.
# xx = Encoder.cast_input_for_init(x);
# s=r(xx,p,s);
# out=r(x,p,s);

################## Test examples ##################################

function encoder_model(; rseed=0)
	rng = Random.default_rng()
	if (rseed != 0) Random.seed!(rng, rseed) end
	#model = Chain(RNN_saf(1,8),RNN_saf(8,4),Dense(4=>1))
	model = Chain(RNN_saf(1,8),RNN_saf(8,4),Dense(4=>6))
	ps,st = Lux.setup(rng, model)
	return model, ps, st
end

function encoder_one_layer_test(; rseed=0)
	rng = Random.default_rng()
	if (rseed != 0) Random.seed!(rng, rseed) end
	model = Chain(RNN_saf(1,8))
	ps,st = Lux.setup(rng, model)
	return model, ps, st
end

################## RNN_saf layer definition ##################################

abstract type AbstractRecurrentCell{use_bias} <: Lux.AbstractLuxLayer end

@concrete struct RNN_saf{use_bias} <: 
				AbstractRecurrentCell{use_bias}
	activation::Function
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias
    init_state
    has_bias::Bool
end

function RNN_saf(in_dims::Int, out_dims::Int, activation=tanh;
		init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32, init_state=Lux.ones32,
		allow_fast_activation::Bool=true, use_bias::Bool=false)
	activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return RNN_saf{use_bias}(activation, in_dims, out_dims, 
		init_weight, init_bias, init_state, use_bias)
end

function Lux.initialparameters(rng::AbstractRNG,
        rnn::RNN_saf{use_bias}) where {use_bias}
    ps = (weight_ih=rnn.init_weight(rng, rnn.out_dims, rnn.in_dims),
        weight_hh=rnn.init_weight(rng, rnn.out_dims, rnn.out_dims),
        hidden_memory=reshape([0.95f0],1,1))
    use_bias && (ps = merge(ps, (bias=rnn.init_bias(rng, rnn.out_dims),)))
    (ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),)))
    return ps
end

function Lux.initialstates(rng::AbstractRNG, ::RNN_saf)
    randn(rng, 1)
    return (rng=Lux.replicate(rng),hidden_state=[])
end

# for parameters, add 1 for cell hidden_memory rate
Lux.parameterlength(l::RNN_saf) = l.out_dims * l.in_dims + l.out_dims^2 + 1 +
									(l.has_bias ? l.out_dims : 0)
Lux.statelength(l::RNN_saf) = l.out_dims

cast_input_for_init(x::AbstractMatrix) = (z -> Float16(z)).(x)

# must give explicit true/false versions to call these functions correctly
(rnn::RNN_saf{true})(x::Matrix{Float16}, ps, st::NamedTuple) =
		local_hidden_state_init(rnn, x, ps, st)
(rnn::RNN_saf{false})(x::Matrix{Float16}, ps, st::NamedTuple) =
		local_hidden_state_init(rnn, x, ps, st)

function local_hidden_state_init(rnn, x::Matrix{Float16}, ps, st::NamedTuple)
    rng = Lux.replicate(st.rng)
    @set! st.rng = rng
    @set! st.hidden_state = _init_hidden_state(rng, rnn, x)
    x = cast_input_for_init(ones(rnn.out_dims, size(x,2)))
    return x, st
end

function (rnn::RNN_saf{true})(x::AbstractMatrix, ps, st::NamedTuple)
    h_new = ps.weight_ih * x .+ ps.weight_hh * st.hidden_state .+ ps.bias
    h_new = ps.hidden_memory[1,1]*st.hidden_state + 
    	(1.0f0-ps.hidden_memory[1,1])*__apply_activation(rnn.activation, h_new)
    @set! st.hidden_state = h_new
    return h_new, st
end

function (rnn::RNN_saf{false})(x::AbstractMatrix, ps, st::NamedTuple)
    h_new = ps.weight_ih * x .+ ps.weight_hh * st.hidden_state
    h_new = ps.hidden_memory[1,1]*st.hidden_state + 
    	(1.0f0-ps.hidden_memory[1,1])*__apply_activation(rnn.activation, h_new)
    @set! st.hidden_state = h_new
    return h_new, st
end

function Base.show(io::IO, r::RNN_saf{use_bias}) where {use_bias}
    print(io, "RNN_saf($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    use_bias || print(io, ", bias=false")
    return print(io, ")")
end

@inline function _init_hidden_state(rng::AbstractRNG, rnn, x::AbstractMatrix)
    return rnn.init_state(rng, rnn.out_dims, size(x, 2))
end

# Activation Function
@inline __apply_activation(::typeof(identity), x) = x
@inline __apply_activation(f, x) = f.(x)
