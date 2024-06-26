using DataFrames, CSV, Plots, Statistics, Flux, Dates

data=CSV.read("/Users/seansantellanes/Documents/RNN/us-epa-pm25-aqi.csv", DataFrame)
data=data[:,Not(2)]

data2=select(data, [2,3],
 AsTable([2,3]) => x
  -> (data[:,2]+data[:,3])/2)
tar_data=select(data2, Not(1,2))
tar_data=rename!(tar_data,[:a])
tar_vector=Vector{Float32}(tar_data[:,:a])
function create_sequences(data, seq_length)
    X, y = [], []
    for i in 1:(length(data) - seq_length)
        push!(X, data[i:i+seq_length-1])
        push!(y, data[i+seq_length])
    end
    return X, y
end

seq_length = 6
X, y = create_sequences(tar_vector, seq_length)
X = hcat(X...)'
y = Float32.(y)
model = Chain(
    LSTM(seq_length, 50),
    LSTM(50,50),
    Dense(50, 1)
)

# Initialize the parameters
ps = Flux.params(model)

# Define the loss function and the optimizer
loss(x, y) = Flux.mse(model(x)[:], y)
opt = ADAM(1e-2)
epochs = 5000
batch_size = seq_length

for epoch in 1:epochs
    
    for i in 1:batch_size:(size(X, 1) - batch_size)
        x_batch = X[i:i+batch_size-1, :]
        y_batch = y[i:i+batch_size-1]
       
        gs = Flux.gradient(ps) do
            loss(x_batch, y_batch)
        end

        Flux.update!(opt, ps, gs)
    end
    #println("Epoch $epoch: Loss = $(loss(X', y))")
end
function predict(model, data, seq_length, num_predictions)
    input = data[end-seq_length+1:end,:]
    pred = model(input)


    return pred
end
num_predictions = 6
predictions = predict(model, X, seq_length, num_predictions)
Float32.(predictions')
typeof(data."DateTime")
format = "yyyy-mm-dd HH:MM:SS"
dt=DateTime.(data."DateTime",format);

for i in 1:seq_length
    local last_dt=dt[end]
    last_dt+=Minute(10)
    push!(dt,last_dt)
end
predictions=convert(Vector{Float32},vec(predictions))
p=plot(dt[1:end-num_predictions], tar_vector, label="Original Data",ylims=(0,60),ylabel="PM2.5",xlabel="Date Time",lw=1.5)
plot!(p,dt[end-num_predictions+1:end], predictions, label="Predictions",color=:black,lw=2.5)
hspan!(p,[0,50],color=:green,alpha=0.2)
hspan!(p,[50,100],color=:yellow,alpha=0.2,legend=false)
