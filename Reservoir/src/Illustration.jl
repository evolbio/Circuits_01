module Illustration
using ReservoirComputing
export illustrate_driver, illustrate_driver_full, normalize!, print_mathematica

# input goes to all nodes
function illustrate_driver_full(input, res_size; weight=0.3)
	esn=ESN(input;
		input_layer=MinimumLayer(),
		reservoir=SimpleCycleReservoir(res_size; weight=weight)
	)
	return esn
end

# input goes only to first node
function illustrate_driver(input, res_size; weight=0.3)
	esn=ESN(input;
		#input_layer=MinimumLayer(),
		input_layer=create_layer(nothing, res_size, size(input)[1]),
		reservoir=SimpleCycleReservoir(res_size; weight=weight)
	)
	return esn
end

function create_layer(nothing, res_size, in_size)
    m=zeros(res_size, in_size)
    m[1,1] = 1.0
    return m
end

function normalize!(x; low = 0, high = 1)
    x .= ((high - low) / (maximum(x) - minimum(x)) .* x)
    x .= x .+ (low - minimum(x))
end

function print_mathematica(matrix)
    rows, cols = size(matrix)
    println("{")
    for i in 1:rows
        print(" {")
        for j in 1:cols
            print(matrix[i,j])
            if j < cols
                print(", ")
            end
        end
        print("}")
        if i < rows
            println(",")
        end
    end
    println("\n}")
end

end # module Illustration
