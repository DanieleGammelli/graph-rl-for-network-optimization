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
 
string path = ...;
{accTuple} availableProd = ...;
{accTuple} desiredShip = ...;
{accTuple} desiredProd = ...;
{accTuple} storageCapacity = ...;
{accTuple} warehouseStock = ...;
{accTuple} demand = ...;

{edgeAttrTuple} edgeAttr = ...;
 
{Edge} edge = {<i,j>|<i,j,t,cost> in edgeAttr};
{int} node = {i|<i,v> in storageCapacity};
{int} node_factory = {i|<i,v> in desiredProd};
{int} node_warehouse = {i|<i,v> in desiredShip};
 
float desiredShipArray[node_warehouse] = [i:v|<i,v> in desiredShip]; // desired total incoming flow at a node
float desiredProdArray[node_factory] = [i:v|<i,v> in desiredProd]; // desired production at a node
float availableProdArray[node_factory] = [i:v|<i,v> in availableProd]; // available product to ship from a node
float capacity[node] = [i:v|<i,v> in storageCapacity]; // storage capacity at a node
float warehouseStockArray[node_warehouse] = [i:v|<i,v> in warehouseStock];
float demandArray[node_warehouse] = [i:v|<i,v> in demand];

dvar float+ flow[edge];
dvar float+ production[node_factory];
dvar float error_f[node_warehouse];
dvar float error_p[node_factory];
dvar float error_cap[node_warehouse];
minimize(sum(i in node_factory)abs(error_p[i]) + sum(i in node_warehouse)abs(error_f[i]) + 10000*sum(i in node_warehouse)abs(error_cap[i]));
subject to
{
  forall(i in node_warehouse)
    {
    sum(e in edge: e.j==i) flow[<e.i, e.j>] == desiredShipArray[i] + error_f[i];
    warehouseStockArray[i] + flow[<0, i>] - demandArray[i] <= capacity[i] + error_cap[i];
    }
    
  forall(i in node_factory)
    {
    sum(e in edge: e.i==i) flow[<e.i, e.j>] <= availableProdArray[i];
    production[i] == desiredProdArray[i] + error_p[i];
    availableProdArray[i] + production[i] - sum(e in edge: e.i==i) flow[<e.i, e.j>] <= capacity[i];
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
         ofile.write(thisOplModel.flow[e]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.write("production=[")
  for(var i in thisOplModel.node_factory)
       {
         ofile.write("(");
         ofile.write(i);
         ofile.write(",");
         ofile.write(thisOplModel.production[i]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.close();
}