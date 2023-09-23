#include "ImageProcessing.h"

Color::Color(int R, int G, int B): R(R), G(G), B(B)
{
}

Color::Color(): R(0), G(0), B(0)
{
}

Pixel::Pixel(int x, int y, int intensity): x(x), y(y), intensity(intensity)
{
}

Pixel::Pixel(): x(0), y(0), intensity(0)
{
}

bool Pixel::operator==(const Pixel& other) const
{
    return (x == other.x && y == other.y);
}

// Edge::Edge(Pixel src, Pixel des, int weight): src(src), des(des), weight(weight)
// {
// }
//
// Edge::Edge(): src(Pixel()), des(Pixel()), weight(0)
// {
// }
//
// Edge* Edge::fromImage(cv::Mat image, int& size)
// {
//     int row = image.rows;
//     int col = image.cols;
//
//     // Pixel array
//     Pixel* pixels = new Pixel[row * col];
//     int index = 0;
//     for (int i = 0; i < row; ++i)
//     {
//         uchar* p = image.ptr<uchar>(i);
//         for (int j = 0; j < col; ++j)
//         {
//             pixels[index++] = Pixel(i, j, p[j]);
//         }
//     }
//
//     Edge* edges = nullptr;
//     index = 0;
//
//     for (int y = 0; y < row; y++)
//     {
//         for (int x = 0; x < col; x++)
//         {
//             // Pixel 1
//             Pixel src = pixels[y * col + x];
//             if (x < col - 1)
//             {
//                 // Last element in row
//                 Pixel des = pixels[y * col + x + 1];
//                 Edge e;
//                 e.src = src;
//                 e.des = des;
//                 e.weight = abs(src.intensity - des.intensity);
//                 index++;
//                 edges = (Edge*)realloc(edges, index * sizeof(Edge));
//                 edges[index - 1] = e;
//             }
//             if (y < row - 1)
//             {
//                 // Last row
//                 Pixel px2 = pixels[(y + 1) * col + x];
//                 Edge e;
//                 e.src = src;
//                 e.des = px2;
//                 e.weight = abs(src.intensity - px2.intensity);
//                 index++;
//                 edges = (Edge*)realloc(edges, index * sizeof(Edge));
//                 edges[index - 1] = e;
//             }
//         }
//     }
//     size = index;
//     return edges;
// }
//
// Edge* Edge::formKruskal(Edge* edges, int col, int row, int& sz)
// {
//     int size = col * row;
//     DisjointSet ds(size);
//     Edge* result = nullptr;
//     int index = 0;
//     for (int i = 0; i < size; ++i)
//     {
//         Pixel src = edges[i].src;
//         Pixel des = edges[i].des;
//         int weight = edges[i].weight;
//         if (ds.find(src.x * col + src.y) != ds.find(des.x * col + des.y))
//         {
//             ds.unionSets(src.x * col + src.y, des.x * col + des.y);
//             index++;
//             result = (Edge*)realloc(result, index * sizeof(Edge));
//             result[index - 1] = edges[i];
//         }
//     }
//
//     // Process MST
//     // Calculate average and standard deviation
//     int sum = 0;
//     for (int i = 0; i < index; ++i)
//     {
//         sum += result[i].weight;
//     }
//     int average = sum / index;
//     int standard_deviation = 0;
//     for (int i = 0; i < index; ++i)
//     {
//         standard_deviation += pow(result[i].weight - average, 2);
//     }
//     standard_deviation = sqrt(standard_deviation / index);
//
//     // Remove outlier
//     Edge* newResult = nullptr;
//     int newIndex = 0;
//     for (int i = 0; i < index; ++i)
//     {
//         if (result[i].weight <= average + standard_deviation)
//         {
//             newIndex++;
//             newResult = (Edge*)realloc(newResult, newIndex * sizeof(Edge));
//             newResult[newIndex - 1] = result[i];
//         }
//     }
//     sz = newIndex;
//     // Sort by weight
//     int counter = 0;
//     for (int i = 0; i < newIndex; i++)
//     {
//         Edge e = newResult[i];
//         if (e.des.x > 83 || e.des.y > 83)
//         {
//             counter++;
//         }
//     }
//
//     for (int i = 0; i < newIndex - 1; ++i)
//     {
//         for (int j = i + 1; j < newIndex; ++j)
//         {
//             if (newResult[i].weight > newResult[j].weight)
//             {
//                 Edge temp = newResult[i];
//                 newResult[i] = newResult[j];
//                 newResult[j] = temp;
//             }
//         }
//     }
//     return newResult;
// }
//
//
void DisjointSet::swap(int& a, int& b)
{
    int temp = a;
    a = b;
    b = temp;
}

DisjointSet::DisjointSet(int size): size(size)
{
    parent = new int[size];
    rank = new int[size];

    for (int i = 0; i < size; ++i)
    {
        parent[i] = i;
        rank[i] = 0;
    }
}

int DisjointSet::find(int v)
{
    if (v == parent[v])
    {
        return v;
    }
    return parent[v] = find(parent[v]);
}

void DisjointSet::unionSets(int a, int b)
{
    a = find(a);
    b = find(b);

    if (a != b)
    {
        if (rank[a] < rank[b])
        {
            swap(a, b);
        }
        parent[b] = a;
        if (rank[a] == rank[b])
        {
            rank[a]++;
        }
    }
}

DisjointSet::~DisjointSet()
{
    delete[] parent;
    delete[] rank;
}
