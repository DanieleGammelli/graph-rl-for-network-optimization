tuple Edge{
  int i;
  int j;
}
 
tuple edgeAttrTuple{
    int i;
    int j;
    int t;
    float cost;
}
 
tuple accTuple{
  int i;
  float n;
}

tuple accTimeTuple{
  int i;
  int t;
  float n;
}

int T = ...;
int prod_time = ...;
string path = ...;
{accTuple} storageCost = ...;
{accTuple} productionCost = ...;
float penalty_cost = ...;
float product_price = ...;
{accTuple} availableProd = ...;
{accTuple} storageCapacity = ...;
{accTuple} warehouseStock = ...;
{accTimeTuple} demand = ...;
{edgeAttrTuple} edgeAttr = ...;
{accTimeTuple} arrivalTuple = ...;
{accTimeTuple} expectedProdTuple = ...;

{Edge} edge = {<i,j>|<i,j,t,cost> in edgeAttr};
{int} node = {i|<i,v> in storageCapacity};
{int} node_factory = {i|<i,v> in availableProd};
{int} node_warehouse = {i|<i,v> in warehouseStock};

float flow_cost[edge] = [<i,j>: cost|<i,j,t,cost> in edgeAttr];
float production_cost[node_factory] = [i:v|<i,v> in productionCost];
float storage_cost[node] = [i:v|<i,v> in storageCost];
int travelTime[edge] = [<i,j>: t|<i,j,t,cost> in edgeAttr];
float prodInit[node_factory] = [i:v|<i,v> in availableProd]; // available product to ship from a node
float capacity[node] = [i:v|<i,v> in storageCapacity]; // storage capacity at a node
float arrival[node][1..T] = [i:[t:v]|<i,t,v> in arrivalTuple ];
float expectedProd[node][1..T] = [i:[t:v]|<i,t,v> in expectedProdTuple ];
float stockInit[node_warehouse] = [i:v|<i,v> in warehouseStock];
float demandArray[node_warehouse][1..T] = [i:[t:v]|<i,t,v> in demand];

dvar float+ flow[edge][1..T];
dvar float+ production[node_factory][1..T];
dvar float stock[node][0..T];
dvar float cost[node][0..T];

maximize(sum(i in node_warehouse, t in 1..T) demandArray[i][t] * product_price 
- sum(i in node,t in 0..T)cost[i][t] - sum(e in edge,t in 1..T) flow[e][t] * flow_cost[e]);
subject to
{
  forall(t in 0..T)
  {
    forall(i in node_warehouse)
    {
      if(t==0)
      	stock[i][t] == stockInit[i];
      else
      {
        stock[i][t] == stock[i][t-1] + sum(e in edge: e.j==i && t - travelTime[e] >= 1)flow[e][t - travelTime[e]] + arrival[i][t] - sum(e in edge: e.i==i)flow[e][t] - demandArray[i][t];
        stock[i][t] <= capacity[i];
      }
      cost[i][t] >= stock[i][t]*storage_cost[i];
      cost[i][t] >= - stock[i][t] * penalty_cost;  
    }
  }
  
  forall(t in 0..T)
  {
    forall(i in node_factory)
    {
      if(t==0)
	 {
      	stock[i][t] == prodInit[i];
		cost[i][t] >= stock[i][t]*storage_cost[i];
		cost[i][t] >= - stock[i][t] * penalty_cost; 
	  }
      else
      {
        if(t - prod_time >= 1)        
        	stock[i][t] == stock[i][t-1] + production[i][t-prod_time] + expectedProd[i][t] - sum(e in edge: e.i==i)flow[e][t];
        else
        	stock[i][t] == stock[i][t-1] + expectedProd[i][t] - sum(e in edge: e.i==i)flow[e][t];
		sum(e in edge: e.i==i)flow[e][t] <= stock[i][t-1];
        stock[i][t] <= capacity[i];
		cost[i][t] >= stock[i][t]*storage_cost[i] + production[i][t]*production_cost[i];
        cost[i][t] >= - stock[i][t] * penalty_cost + production[i][t]*production_cost[i]; 
      }
      
    }
  }
}
 
 
main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.edge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.flow[e][1]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.write("production=[")
  for(var i in thisOplModel.node_factory)
       {
         ofile.write("(");
         ofile.write(i);
         ofile.write(",");
         ofile.write(thisOplModel.production[i][1]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.close();
}