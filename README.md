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

> Split dataset size 0.2
```python
train['difficulty'] = train['difficulty'] / 10000000000
# 10,000,000,000 divid
  
columns = ['miners-revenue', 'trade-volume',
        'n-unique-addresses']
        
x_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]


from sklearn.model_selection import train_test_split
train, test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train = pd.concat([train,y_train],axis=1)
y_test = y_test/1000000
train['hash-rate'] = train['hash-rate']/1000000
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
class NN(torch.nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        
        self.linear1 = nn.Linear(6,512,bias=True)
        self.linear2 = nn.Linear(512,256,bias=True)
        self.linear3 = nn.Linear(256,128,bias=True)
        self.linear4 = nn.Linear(128,64,bias=True)
        self.linear5 = nn.Linear(64,32,bias=True)
        self.linear6 = nn.Linear(32,1,bias=True)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.1)
        #init
        
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.orthogonal_(self.linear3.weight)
        torch.nn.init.orthogonal_(self.linear4.weight)
        torch.nn.init.orthogonal_(self.linear5.weight)
        torch.nn.init.orthogonal_(self.linear6.weight)
        
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear3(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear4(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear5(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear6(out)
        return out

model = NN().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)
loss = nn.MSELoss().to(device)
model
```
>NN->(  
>  (linear1): Linear(in_features=6, out_features=512, bias=True)  
>  (linear2): Linear(in_features=512, out_features=256, bias=True)  
>  (linear3): Linear(in_features=256, out_features=128, bias=True)  
>  (linear4): Linear(in_features=128, out_features=64, bias=True)  
>  (linear5): Linear(in_features=64, out_features=32, bias=True)  
>  (linear6): Linear(in_features=32, out_features=1, bias=True)  
>  (relu): ReLU()  
>)  


# Train
```python
plt_los = []
train_total_batch = len(x_train)
for epoch in range(10001) : 
    avg_cost = 0
    model.train()

    hypothesis = model(x_train) 
    cost = loss(hypothesis,y_train) 
    
    optimizer.zero_grad() 
    cost.backward()
    optimizer.step() 
    avg_cost += cost / train_total_batch
    plt_los.append([cost])
    if epoch%100==0:  
        print("Epoch : ",epoch,"cost : ",cost.item())
```

# Loss
```python
import matplotlib.pyplot as plt

def plot(loss_list: list, ylim=None, title=None) -> None:
    bn = [i[0] for i in loss_list]

    plt.figure(figsize=(7, 7))
    plt.plot(bn, label='train')
    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)
    plt.legend()
    plt.grid('on')
    plt.show()
 
plot(plt_los , [30.0, 100.0], title='Loss at Epoch')
```
![Screenshot 2021-11-09 200657](https://user-images.githubusercontent.com/82564045/140913381-682d9dfc-4070-4355-aebd-259f1a0ba9b1.gif)
