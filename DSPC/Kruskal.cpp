// #include <iostream>
// #include <vector>
// #include <unordered_map>
// #include <algorithm>
//
// using namespace std;
// // Define the DIRECTIONS array as a global constant
// const vector<pair<int, int>> DIRECTIONS = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
//
// class UnionFind
// {
// public:
//     UnionFind(const vector<pair<int, int>>& vertices, const vector<int>& values)
//     {
//         for (size_t i = 0; i < vertices.size(); i++)
//         {
//             parent[vertices[i]] = vertices[i];
//             rank[vertices[i]] = 0;
//             sizes[vertices[i]] = 1;
//             min_values[vertices[i]] = values[i];
//             max_values[vertices[i]] = values[i];
//         }
//     }
//
//     pair<int, int> find(const pair<int, int>& x)
//     {
//         if (parent[x] == x)
//             return x;
//         return find(parent[x]);
//     }
//
//     void union_sets(const pair<int, int>& x, const pair<int, int>& y)
//     {
//         pair<int, int> x_root = find(x);
//         pair<int, int> y_root = find(y);
//
//         if (rank[x_root] < rank[y_root])
//         {
//             parent[x_root] = y_root;
//             sizes[y_root] += sizes[x_root];
//             min_values[y_root] = min(min_values[y_root], min_values[x_root]);
//             max_values[y_root] = max(max_values[y_root], max_values[x_root]);
//         }
//         else
//         {
//             parent[y_root] = x_root;
//             sizes[x_root] += sizes[y_root];
//             min_values[x_root] = min(min_values[x_root], min_values[y_root]);
//             max_values[x_root] = max(max_values[x_root], max_values[y_root]);
//             if (rank[x_root] == rank[y_root])
//                 rank[x_root]++;
//         }
//     }
//
//     int size(const pair<int, int>& x)
//     {
//         return sizes[find(x)];
//     }
//
//     int max_diff(const pair<int, int>& x)
//     {
//         return max_values[find(x)] - min_values[find(x)];
//     }
//
//     unordered_map<pair<int, int>, pair<int, int>> parent;
//     unordered_map<pair<int, int>, int> rank;
//     unordered_map<pair<int, int>, int> sizes;
//     unordered_map<pair<int, int>, int> min_values;
//     unordered_map<pair<int, int>, int> max_values;
// };
//
// bool is_valid_position(int next_x, int next_y, int n, int m)
// {
//     return (next_x >= 0 && next_x < n) && (next_y >= 0 && next_y < m);
// }
//
// vector<pair<int, pair<int, int>>> get_edges(const vector<vector<int>>& image, int n, int m)
// {
//     vector<pair<int, pair<int, int>>> edges;
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < m; j++)
//         {
//             for (const auto& dir : DIRECTIONS)
//             {
//                 int next_x = i + dir.first;
//                 int next_y = j + dir.second;
//                 if (!is_valid_position(next_x, next_y, n, m))
//                     continue;
//                 int weight = abs(image[i][j] - image[next_x][next_y]);
//                 edges.push_back({weight, {i, j}, {next_x, next_y}});
//             }
//         }
//     }
//     return edges;
// }
//
// vector<pair<int, int>> get_roots(const UnionFind& uf)
// {
//     vector<pair<int, int>> roots;
//     for (const auto& entry : uf.parent)
//     {
//         if (entry.first == entry.second)
//             roots.push_back(entry.first);
//     }
//     return roots;
// }
//
// void label_connected_component(int i, int j, UnionFind& uf, int label, vector<vector<int>>& res)
// {
//     pair<int, int> curr = {i, j};
//     while (uf.find(curr) != curr)
//     {
//         res[curr.first][curr.second] = label;
//         curr = uf.find(curr);
//     }
//     res[curr.first][curr.second] = label;
// }
//
// vector<vector<int>> segment_image(int k, const vector<vector<int>>& image)
// {
//     int n = image.size();
//     int m = image[0].size();
//
//     vector<pair<int, int>> vertices;
//     vector<int> values;
//
//     // Get vertices and values to build UnionFind
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < m; j++)
//         {
//             vertices.push_back({i, j});
//             values.push_back(image[i][j]);
//         }
//     }
//
//     UnionFind uf(vertices, values);
//
//     // Get all edges and sort by weight
//     vector<pair<int, pair<int, int>>> edges = get_edges(image, n, m);
//     sort(edges.begin(), edges.end());
//
//     // Kruskal's algorithm
//     for (const auto& edge : edges)
//     {
//         int weight = edge.first;vousing using namespace cv;// To Convert an image to con vert an image to graph to be used by MSTstruct Pixel {}int x,y, , weight;int x;int y;int weight;intensitystruct vector<EPixel> convertMatToPixel(Mat image_) {}
//         pair<int, int> u = edge.second;
//         pair<int, int> v = edge.third;
//
//         pair<int, int> u_root = uf.find(u);
//         pair<int, int> v_root = uf.find(v);
//         int threshold = min(uf.max_diff(u) + k / uf.size(u), uf.max_diff(v) + k / uf.size(v));
//
//         if (u_root != v_root && weight <= threshold)
//         {
//             uf.union_sets(u, v);
//         }
//     }
//
//     // Initialize the result vector
//     vector<vector<int>> res(n, vector<int>(m, -1));
//
//     // Get all roots (the ultimate parent of each segment)
//     vector<pair<int, int>> roots = get_roots(uf);
//
//     // Assign a unique label to each segment
//     unordered_map<pair<int, int>, int> root_to_label;
//     int label = 0;
//     for (const auto& root : roots)
//     {
//         root_to_label[root] = label;
//         label++;
//     }
//
//     // Label all connected components
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < m; j++)
//         {
//             if (res[i][j] != -1)
//                 continue;
//             pair<int, int> curr_root = uf.find({i, j});
//             int curr_label = root_to_label[curr_root];
//             label_connected_component(i, j, uf, curr_label, res);
//         }
//     }
//
//     return res;
// }
//
// int main()
// {
//     // Example usage
//     int k = 10;
//     vector<vector<int>> image = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//     vector<vector<int>> segmented_image = segment_image(k, image);
//
//     // Output the segmented image
//     for (const auto& row : segmented_image)
//     {
//         for (int label : row)
//         {
//             cout << label << " ";
//         }
//         cout << endl;
//     }
//
//     return 0;
// }
