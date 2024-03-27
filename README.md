## 待学习

树状dp，树状数组，线段树，替罪羊树，根号数组，模拟退火，c++stl

# 算论



## 欧拉筛

获取范围内的质数

```java
int n;
int[] prime=new int[n+1];
boolean[] notPrime=new boolean[n+1];
notPrime[0]=true;notPrime[1]=true;
//0,1不是素数
int cnt=0;
for(int i=2;i<=n;i++){
    if(notPrime[i]==false) prime[cnt++]=i;
    // 如果i没有被前面的数筛掉，则i是素数
    for(int j=1; j<=cnt && i*prime[j]<=n ;j++){
        // 筛掉i的素数倍，即i的prime[j]倍
        notPrime[i*prime[j]]=true;
        if(i%prime[j]==0) break;//error:prime[j]==0
        // 如果i整除prime[j]，退出循环，保证线性的时间复杂度
    }
}
```

## 埃氏筛

获取范围内的质数

```java
boolean[] notPrime=new boolean[n+1];
notPrime[0]=true;notPrime[1]=true;
for(int i=2;i<=Math.sqrt(n);i++){
	if (notPrime[i]==false){//i是素数
		for(int j=i*i;j<=n;j+=i){
            //筛掉i的倍数
            notPrime[j]=true;
         }
    }
}
```

## 质因数分解

```java
int num;
List<Integer> ans=new ArrayList<>();
for(int i=2;i<=Math.sqrt(num);i++){
    while(num%i==0){//非质数的因数必先被分解
        ans.add(i);
        num/=i;
    }
    if(num<=1) break;
}
if(num>1) ans.add(num);
return ans;
```

## 最大公约数gcd

```java
int gcd(int a,int b){
    int temp=0;
    while(b!=0){
        temp=b;
        b=a%b;
        a=temp;
    }
    return a;
}
//最小公倍数
int c=a*b/gcd(a,b);
```



# 算法



## 二分搜索

用于在顺序数组搜索目标，时间复杂度O(log2 n)

```java
int search(int[] nums, int target) {
    int left =0, right = nums.length; //搜索区间[left,right)
    while(left<right){
        int mid = left + (right - left)/2;
        if(nums[mid] < target)
            left = mid+1;
        else
            right = mid;
    }
    return left;//lower_bound 返回 大于或等于 target的第一个元素位置
    //upper_bound 改nums[mid]<=target 返回 大于 tar的第一个元素位置
}
```

```java
Arrays.binarySearch(int[] arr, int target);
```





## 动态规划

### 背包DP

#### 0-1背包

背包容量(重量)限制下选取物品获得最大的价值，每种物品**最多选一次**

**二维数组**解法(朴素)：

```java
//1.确定数组维度定义
int[][] dp= new int[i][j]//i表示在物品已经选取范围0~i，j表示目前背包体积
//2.确定转移方程
for(int i=0;i<items.length;i++){
    for(int j=1;j<=max_volume;j++){
        dp[i][j]=Math.max( dp[i-1][j] , dp[i-1][j-volume[i]]+val[i] );
    }			/*不选i,操作前后体积和价值不变*/	/*选i，操作前后体积变化volumn[i],价值变化val[i]*/
}
//3.确定初始值
  \j
 i \ 0 1 2 3
  0  0 0 4 4
  1  0
  2  0
//观察转移方程，每状态由上方或左上方状态得来，必须初始化最左列和最上行      
dp[i][0]=0//体积为0时必定无物品选取
dp[0][j] = j>=volumn[0]?val[0]:0;//选取范围物品0，j体积下最大价值

```

**一维数组**解法(状态压缩)：

```java
 /* 由二维数组解法简化而来，空间复杂度低一度
	由于数组dp第i层由上一层推出
	因此可以只要一层dp数组
	层内遍历由高位向低位推进（右到左），保证本次遍历的结果完全由上次遍历得到 */
dp[0]=0;
for(int i=0;i<items.length;i++){//遍历物品
    for(int j=max_volume;j>=volume[i];j--){//遍历容量(逆序)
        dp[j]=Math.max( dp[j] , dp[j-volume[i]]+val[i] );
    }
}
```



#### 多重背包

背包容量(重量)限制下选取物品获得最大的价值，每种物品**最多选num次**

```java
/*  与01背包相似
	把每种物品的个数放在01背包里面在遍历一遍 */
for(int i=0;i<items.length;i++){//遍历物品
    for(int j=max_volume;j>=volume[i];j--){//遍历容量
        for(int k=1;k<=num[i];k++){//遍历物品数量
            dp[j]=Math.max( dp[j] , dp[j-k*volume[i]]+k*val[i] );
        }
    }
}
```



#### 完全背包

背包容量(重量)限制下选取物品获得最大的价值，每种物品**选择次数不限**

```java
/* 层内遍历由低位向高位推进（左到右），因为可重复选，本次遍历的结果可由本次遍历先前结果得到
  （对于物品i，容量小时选过，容量大时可再选）*/
for(int i=0;i<items.length;i++){//遍历物品
    for(int j=volume[i];j<=max_volume;j++){//遍历容量
        dp[j]=Math.max( dp[j] , dp[j-volume[i]]+val[i] );
    }
}
```

完全背包变种：为得到容量v的物品总量选择物品，求物品的组合方式数/排列方式数

```java
//组合方式数,物品按item内顺序出现
dp[0]=1;/**/
for(int i=0;i<items.length;i++){//先遍历物品
    for(int j=volume[i];j<=max_volume;j++){//再遍历容量
        dp[j]+=dp[j-volume[i]];
    }
}
//排列方式数,物品出现顺序无要求
for(int j=0;j<=max_volume;j++){//先遍历容量
    for(int i=0;i<items.length;i++){//再遍历物品
		if(j-volume[i]>=0)
			dp[j]+=dp[j-volume[i]];
    }
}
```



#### 分组背包

背包容量(重量)限制下在各组选取物品获得最大的价值，**每组物品最多选一个**

```java
/*  分组背包问题实际上与的01背包十分相似
	区别在于分组背包DP状态数组第一维表示的是，只考虑第 1 ~ i 组*/
for(int i = 1; i <= n; i++){//遍历分组
    for(int j = m; j >= 0; j--){//遍历容量(逆序)
        for(int k = 0; k < s[i]; k++)//遍历组内物品
            if(j-v[i][k] >= 0)
                dp[j] = Math.max( dp[j], dp[j-volume[i][k]] + val[i][k]);
    	}
	}
}
```



### 区间DP

所谓**区间dp**，指在一段区间上进行动态规划，一般做法是由长度较小的区间往长度较大的区间进行递推，最终得到整个区间的答案，而边界就是长度为1以及2的区间。

区间dp常见的转移方程如下：

```java
int dp[i][j] = min{ dp[i][k] + dp[k+1][j]} + w(i,j) };   (i <= k < j)
```

其中`dp[i][j]`表示在区间`[i,j]`上的最优值，`w(i,j)`表示在转移时需要额外付出的代价，min也可以是max。

为确保先得到子区段，保证父区段结果正确，一般按区段长度j-i递增顺序计算各区段值



### 树形DP

由于树固有的递归性质，树形 DP 一般都是递归(DFS)进行的。以各个节点为根，依据条件设置不同状态的返回值



## BFS广度优先搜索

在树中是层序遍历，耗时方差小于DFS

核心代码:

```java
//计算从起点start到终点target的最近距离
int BFS(Node start , Node target){
    Queue<Node> q;//结点队列
    Set<Node> visited;//记录已遍历结点，避免走回头路
    q.offer(start);//起点入列
    visited.add(start);
    int step = 0;//记录扩散的步数
    
    while(q not empty){
        int size=q.size();
        /*扩散当前队列的所有结点*/
        for(int i=0;i<size;i++){
            Node cur = q.poll();
            if(cur is target)//判断是否到达终点
                return step;
            for(Node x:cur.adj()){/*cur结点的相邻结点入列*/
                if(x not in visited){
                    q.offer(x);
                    visited.add(x);
                }
            }
        }
        step++;//!更新步数
    }
}
```



## Floyd - Warshall最短路径

解决带权图的多源(任意两节点间)最短路径问题，权重可以为负，它是一种动态规划

核心思想

> **以每个点为「中转站」，刷新邻接点间的距离** 

时间复杂度O(n^3)

```java
public static void floydWarshall(int[][] G) {
    int i, j, k;
    int n=G.length;
    int[][] P=new int[n][n];// 记录各个顶点之间的最短路径
    for (k = 0; k < n; k++) {//以k为中间节点，遍历所有节点组合
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                // 如果经过k节点距离更短，则更新 graph 数组
                if (G[i][j] > G[i][k] + G[k][j]) {
                    G[i][j] = G[i][k] + G[k][j];
                    P[i][j] = k;/* 记录路径:两节点间最短路径经过k
                    			  打印最短路径:i经k到j，i到k，k到j之间可能有其他节点，递归搜索 */
                }
            }
        }
    }
}
```



## Dijkstra最短路径

解决带权图的单源(一个源点到其他节点)最短路径问题，权重不能为负，它是一种贪心算法

核心思想

> **1. 选定一个点，这个点满足两个条件：1.未被选过，2.距离最短**
>
> **2. 对于这个点的所有邻近点去尝试松弛** 

时间复杂度O(n^2)，dis用**优先队列**优化后O(nlogn)

```java
final int inf=9000;//inf代表无穷，要大于最大结果，但注意与某段边长相加后不能溢出
public int[] dijkstra(int[][] G,int source){//G为n*n的图,source为源点
    
    int n=G.length;//n代表图的顶点个数
    int[] dis=new int[n];//dis代表源点到其它点的最短距离，inf代表无穷
    Arrays.fill(dis,inf);//dis初始化为无穷
    dis[source]=0;//源点到源点的距离为0
    boolean[] vis=new boolean[n];//vis代表某个顶点是否被访问过
    
    //使用一个for循环，循环n次，来寻找n个点到源点的最短距离
    for(int i=0;i<n;i++){
        int node=-1;//没有被访问过且距离源点最短的点
        for(int j=0;j<n;j++){
            if(!vis[j]&&(node==-1||dis[j]<dis[node])){
                node=j;//找到当前距离源点最短距离的点node
            }
        }
        //对这个距离源点最短距离的点的所有邻接点进行松弛
        for(int j=0;j<n;j++){
			dis[j]=Math.min(dis[j],dis[node]+G[node][j]);
            /*注意：对于不是node的邻接点并不会影响它原来的距离
                   对于邻接的已经访问过的点也不会产生影响 */
        }
        vis[node]=true//标记为已访问过
    }
    return dis;//返回所有点到源点的最短距离
}
```



## Bellman-Ford最短路径

解决带权图的单源最短路径问题，比Dijkstra算法更具普遍性，对边没有要求，可以处理负权边与负权回路。

核心思想 

> **对所有的边进行n-1轮松弛操作**

时间复杂度高达O(VE), V为顶点数，E为边数

```java
public int[] Bellman(Edge[] edges,int source){
    int[] dis=new int[n];
    Arrays.fill(dis,inf);
    distance[source]=0;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < m; j++) {//对m条边进行循环
            Edge edge = edges[j];
            // 松弛操作
            if (dis[edge.to] > dis[edge.from] + edge.weight ){ 
                dis[edge.to] = dis[edge.from] + edge.weight;
                //如果一条边的末端点到源点距离 大于 始端点到源点的距离加边长，则对末端点到源点距离松弛
            }
        }
    }
    return dis;
}
```



## 滑动窗口

​	滑动窗口算法是在给定特定窗口大小的数组或字符串上执行要求的操作，它将一部分问题中的嵌套循环转变为一个单循环，因此它可以减少时间复杂度。滑动窗口算法更多的是一种思想，而非某种数据结构的使用。

​	可以用来解决一些**查找满足一定条件的连续区间的性质**（长度等）的问题。由于区间连续，因此当区间发生变化时，可以通过旧有的计算结果对搜索空间进行剪枝，这样便减少了重复计算，降低了时间复杂度。往往类似于“ 请找到满足 xx 的最 x 的区间（子串、子数组）的 xx ”这类问题都可以使用该方法进行解决。

核心思想

> **滑动：**这个窗口是按照一定方向移动的
>
> **窗口：**窗口大小并不是固定的，可以不断扩容直到满足一定的条件；也可以不断缩小，直到找到一个满足条件的最小窗口；当然也可以是固定大小。

思路：

1. 我们在字符串 S 中使用双指针中的左右指针技巧，初始化 left = right = 0，把索引闭区间 [left, right] 称为一个「窗口」。
2. 我们先不断地增加 right 指针扩大窗口 [left, right]，直到窗口中的字符串符合要求（包含了 T 中的所有字符）。
3. 此时，我们停止增加 right，转而不断增加 left 指针缩小窗口 [left, right]，直到窗口中的字符串不再符合要求（不包含 T 中的所有字符了）。同时，每次增加 left，我们都要更新一轮结果。
4. 重复第 2 和第 3 步，直到 right 到达字符串 S 的尽头。



# 数据结构



## 单调栈

单调栈可以在时间复杂度为 O(n) 的情况下，求解出某个元素左边或者右边第一个比它大或者小的元素。

所以单调栈一般用于解决一下几种问题：

寻找左 / 右侧第一个比当前元素大 / 小的元素。




## 单调队列

单调递增或递减的队列，用于解决得到当前某个范围（窗口）内的最小值或最大值的问题

核心思想

> 维护一个双向队列（deque），遍历序列，仅当一个元素**可能**成为某个区间最值时才保留它。
>
> 如果维护区间最小值，那么维护的队列就是单调递增的。
>
> 如果维护区间最大值，那么维护的队列就是单调递减的。

队列可存放元素或元素的下标，判断超出窗口元素的方式有不同

```java
//本示例队列存放下标，单调递减，维护最大值
Deque<Integer> deque=new LinkedList<>();
for(int i=0;i<n;i++){//i即窗口右界，窗口大小为m
    if(!deque.isEmpty()&&i-deque.peekFirst()>=m){//移除超出窗口的元素
        deque.removeFirst();
    }
    while(!deque.isEmpty()&&value[i]>value[deque.peekLast()]){//移除所有比新元素小的队尾元素
        deque.removeLast();
	}
    deque.addLast(i);//新元素入列
    System.out.println("窗口中最大元素:"+value[deque.peekFirst()]);
}
```



## 堆(优先队列)

- 堆中某个结点的值总是不大于或不小于其父结点的值；
- 堆总是一棵完全二叉树。

​	将根结点最大的堆叫做最大堆或大根堆，根结点最小的堆叫做最小堆或小根堆。优先队列具有最高级先出的行为特征，通常采用堆数据结构来实现。

​	Java PriorityQueue 没有任何参数的优先级队列默认为小顶堆，可以实现Comparator 接口自定义元素的顺序

```java
//创建自然排序的优先队列
PriorityQueue<Integer> q = new PriorityQueue<Integer>();
//入列
q.add(1);//如果队列已满，则会引发异常
q.offer(2);//如果队列已满，则返回false
//出列
System.out.println(q.poll());  //1，返回并删除队列的开头
System.out.println(q.peek());  //2，返回队列的开头

//创建自定义排序的优先队列
PriorityQueue<Student> q = new PriorityQueue<Student>((a,b)->b.val-a.val);
```

堆排序(构建大顶堆，输出最大值到末尾)

```java
void buildMaxHeap(int[] arr){//arr[0]无元素，根在arr[1]
    for(int i=len/2;i>0;i--){//自下而上遍历非叶子结点
        HeadAdjust(arr,i,arr.length);//将以i为结点的子树进行调整
    }
}
void HeadAdjust(int[] arr,int k,int len){//O(logn)
    arr[0]=arr[k];
    for(int i=2*k;i<=len;i*=2){
        if(i<len&&arr[i]<arr[i+1]){
            i++;//选取较大的子节点下标
        }
        if(arr[0]>=arr[i]){
            break;//结束筛选
        }else{
            arr[k]=arr[i];//较大的子节点向上调整
            k=i;//继续向下筛选
        }
    }
    arr[k]=arr[0];
}
void HeapSort(int[] arr){
	buildMaxHeap(arr);//建堆
    for(int i=len;i>1;i--){
        swap(arr[0],arr[i]);//输出堆顶
        HeadAdjust(arr,1,i-1);//调整剩余i-1个元素
    }
}
```



## 并查集

并查集支持两种操作：

- 查询（Find）：查询某个元素所属集合（查询对应的树的根节点），这可以用于判断两个元素是否属于同一集合

- 合并（Union）：合并两个元素所属集合（合并对应的树）

  ```java
  int[] map=new int[length]; //所有节点，下标表示节点，存放父节点下标
  Arrays.fill(map,-1);//开始所有节点是根节点，根节点内容为-1    
  
  public int find(int[] map,int x){//返回x的根节点
      int p=x;
      while(map[p]!=-1){//找出根节点p
          p=map[p];
      }
      int n1=x,n2=x;
      while(n1!=p){//路径压缩，使路径上的所有结点指向根结点
          n1=map[n1];//n1指向父节点
          map[n2]=p;//n2指向根节点
          n2=n1;//n2尾随n1
      }
      return p;
  }
  
  public void union(int[] map,int x,int y){
      int xRoot=find(map,x);
      int yRoot=find(map,y);
      if(xRoot==yRoot) return;//同根不合并
      map[yRoot]=xRoot;
  }
  ```



## 字典树

trie，利用字符串的公共前缀,在存储时节约存储空间,并在查询时最大限度的减少无谓的字符串比较.

作用：

​	1.以最节约空间的方式存储大量字符串.且存好后是有序的
​           因为是有序的,故而字典树不仅可用于大量字符串的存储,还可用于大量字符串的排序.

​	2.快速查询某字符串s在字典树中是否已存在,甚至出现过几次
  		因为当字典树预处理好之后,查询字符串s在当前的出现情况的效率为strlen(s),异常高效,故而常用于搜索引擎等.

```java
class TrieNode{
    int count;
    int prefix;
    TrieNode[] nextNode;
    public TrieNode(){
        count=0;//当前单词数
        prefix=0;//以当前单词为前缀的单词数
        nextNode=new TrieNode[26];
    }
}
public void insert(String str,TrieNode root){
    if(root==null||str.length()==0) return;
    char[] c=str.toCharArray();
    for(int i=0;i<str.length();i++){
    	if(root.nextNode[c[i]-'a']==null){
    		root.nextNode[c[i]-'a']=new TrieNode();
    	}
    	root=root.nextNode[c[i]-'a'];
    	root.prefix++;
    }
    root.count++;
}
public boolean exist(String str,TrieNode root){
    if(root==null||str.length()==0) return false;
    char[] c=str.toCharArray();
    for(int i=0;i<str.length();i++){
    	if(root.nextNode[c[i]-'a']==null) return false;
    	else root=root.nextNode[c[i]-'a'];
    }
    return root.count>0;
}
```



## 树状数组

树状数组是一种支持 **单点修改** 和 **区间和查询** 的，代码量小的数据结构。

查询和更新的时间复杂度为O(logn)

```java
class Fenwick {
    private final int[] tree;
    public Fenwick(int n) {
        tree = new int[n+1];
    }
    // 把下标为 i 的元素增加 x , i>0
    public void add(int index, int x) {
        for(int i=index+1; i<tree.length; i += i&-i) {
            tree[i] += x;
        }
    }
    // 返回下标在 [0,i) 的元素之和
    public int pre(int index) {
        int res = 0;
        for(int i=index; i>0; i -= i&-i){
            res += tree[i];
        }
        return res;
    }
}

```

