# Hash Power / Hash Rate
What Is Hash Power/Hash Rate?
Hash power, or hash rate, are interchangeable terms used to describe the combined computational power of a specific cryptocurrency network or the power of an individual mining rig on that network.  
The operation of any mineable cryptocurrency — for example, Bitcoin (BTC) — is maintained by its own network of miners: individuals and organizations that contribute the computational power of their mining rigs to process transactions and emit new coins.  They do this via calculating cryptographic hashes — pseudorandom data strings that are used to prevent double spending and to ensure that new coins can’t be created out of thin air.  
>The hash rate of a mining rig is the number of hashes that it can calculate per second. The combined hash power of a cryptocurrency network is the sum of the hash rates of all mining rigs that are in operation at any given moment.

Different devices, such as CPUs, GPUs and ASICs have differing hash rates, depending on their sheer computational power, as well as how well-optimized they are for the specific task of processing a given hash function.
The hash rate of an individual device is a key metric for measuring the profitability of a mining setup as it determines the likelihood of finding a “good” hash that will produce a mining reward. 
On the other hand, the overall hash rate of a cryptocurrency network is an indicator of that coin’s security: in order to hack the network for personal gain, the attackers need to overcome its total hash power — making the task nearly impossible at high enough hash rates.

# DataSets - [BlockChain_BitCoin](https://www.blockchain.com/charts)
- payaments-per-block
- miners-revenue
- trade-volume
- avg-block-size
- unique-addresses
- difficulty
- hash-rate

|Timestamp|n-payments-per-block|miners-revenue|trade-volume|avg-block-size|n-unique-addresses|difficulty|hash-rate|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|11/1/2018 0:00|2851|12416572|232850456|0.979455208|505058|7.18E+12|51977956|
|11/2/2018 0:00|3487|11791349|198797396|1.11705493|565269|7.18E+12|52845282|
|11/3/2018 0:00|2448|12840789|191863094|0.885550503|453208|7.18E+12|52846859|
|...|...|...|...|...|...|...|...|

# Processing

> Split dataset size 0.33
```python
from sklearn.model_selection import train_test_split
train, test, y_train, y_test = train_test_split(
                x_train, y_train,
                test_size=0.33, 
                random_state=42
               )
```

> difficulty scale processing
```python
train['difficulty'] = train['difficulty'] / 10000000000

columns = ['n-payments-per-block', 'miners-revenue', 'trade-volume',
        'n-unique-addresses', 'difficulty', 'hash-rate']

# float -> int
for i in columns:
  train[i] = train[i].astype('int')
```


# Model
```python
linear1 = nn.Linear(7,64,bias=True) # train Feature => 7
linear2 = nn.Linear(64,128,bias=True)  
linear3 = nn.Linear(128,128,bias=True)  
linear4 = nn.Linear(128,64,bias=True)  
linear5 = nn.Linear(64,1,bias=True)  
relu = nn.ReLU() # use ReLU
dropout = nn.Dropout(p=0.3)

#torch.nn.init.xavier_normal_(linear1.weight)

model = nn.Sequential(linear1,relu,dropout,
                      linear2,relu,dropout,
                      linear3,relu,dropout,
                      linear4,relu,dropout,
                      linear5)

optimizer = optim.Adam(model.parameters(), lr=0.1)
loss = nn.MSELoss()

for epoch in range(101) :
    hypothesis = model(x_train)  # set hypothesis
    cost = loss(hypothesis,y_train)
    
    optimizer.zero_grad() # reset parameters
    cost.backward()  # back propagation cost value
    optimizer.step() #optimizer update
    
    if epoch%100==0:  # print cost per epoch 100
        print("Epoch : ",epoch,"cost : ",cost.item())
        
```
