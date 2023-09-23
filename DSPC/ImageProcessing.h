#pragma once
#include <opencv2/core/mat.hpp>

class Color
{
public:
    int R;
    int G;
    int B;

    Color(int R, int G, int B);

    Color();
};

class Pixel
{
public:
    int x;
    int y;
    int intensity;

    Pixel(int x, int y, int intensity);

    Pixel();

    bool operator==(const Pixel& other) const;
};

// class Edge
// {
// public:
//     Pixel src;
//     Pixel des;
//     int weight;
//
//     Edge(Pixel src, Pixel des, int weight);
//
//     Edge();
//
//     static Edge* fromImage(cv::Mat image, int& size);
//     
//     static Edge* formKruskal(Edge* edges, int col, int row, int& size);
//
// };
//
//
class DisjointSet
{
private:
    void swap(int& a, int& b);

public:
    DisjointSet(int size);

    int find(int v);

    void unionSets(int a, int b);

    ~DisjointSet();

private:
    int* parent;
    int* rank;
    int size;
};
