// typedef struct edge
// {
//     unsigned int src;
//     unsigned int des;
//     unsigned int weight;
// } Edge;
//
// typedef struct Pixel
// {
//     unsigned int x;
//     unsigned int y;
//     unsigned int intensity;
// };
//
// typedef struct disjointSet
// {
//     unsigned int* parent;
//     unsigned int* rank;
//     unsigned int size;
// } DisjointSet;
//
// void swap(int& a, int& b)
// {
//     int temp = a;
//     a = b;
//     b = temp;
// }
//
// int find(DisjointSet disjointSet, int v)
// {
//     if (v == disjointSet.parent[v])
//     {
//         return v;
//     }
//     return disjointSet.parent[v] = find(disjointSet, disjointSet.parent[v]);
// }
//
// void unionSets(int a, int b, DisjointSet disjointSet)
// {
//     a = find(disjointSet, a);
//     b = find(disjointSet, b);
//
//     if (a != b)
//     {
//         if (disjointSet.rank[a] < disjointSet.rank[b])
//         {
//             swap(a, b);
//         }
//         disjointSet.parent[b] = a;
//         if (disjointSet.rank[a] == disjointSet.rank[b])
//         {
//             disjointSet.rank[a]++;
//         }
//     }
// }
//
// {
//     
// }
//
//
// __kernel
//
// void processEdges(__global Edge* edges,
//                   __global uchar* segmented,
//                   __global DisjointSet* disjointSets,
//                   unsigned int numEdges,
//                   unsigned int numCols)
// {
//     int gid = get_global_id(0);
//
//     if (gid < numEdges)
//     {
//         Edge edge = edges[gid];
//         DisjointSet disjointSet = disjointSets[0]; // Assuming only one disjoint set in this example
//
//         int parentA = disjointSet.parent[edge.src];
//         int parentB = disjointSet.parent[edge.des];
//
//         if (parentA != parentB)
//         {
//             segmented[edge.src / numCols * numCols + edge.src % numCols] = 255; // Assign segment label
//             // Union operation (assuming the unionSets function is defined in the host code)
//             unionSets(disjointSet, parentA, parentB);
//         }
//     }
// }
