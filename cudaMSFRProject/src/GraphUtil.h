#pragma once
#include "ObjMesh.h"

#include <memory>
#include <algorithm>
#include <time.h>

class Edge {
public:
  int start_, end_;
  double dist;

  Edge():start_(-1), end_(-1), dist(0) {}
  Edge(const Edge & e) :start_(e.start_), end_(e.end_), dist(e.dist) {}
  //Edge &operator=(const Edge& e) : start_(e.start_), end_(e.end_), dist(e.dist) { return *this; }
  Edge(int start, int end) : start_(start), end_(end), dist(0) {}

  
};

class EdgeSet {
public:
  EdgeSet(){}
  EdgeSet(const ObjMesh &obj);
  auto begin(int index) { return head_[index]; }
  auto end(int index) { return end_[index]; }
  const int edges_num() const { return edges_num_; }

private:
  int vertices_num_;
  int edges_num_;
  std::vector<Edge> edge_set_;
  std::vector<std::vector<Edge>::iterator> head_;
  std::vector<std::vector<Edge>::iterator> end_;
};

class GraphUtil :public ObjMesh {

public:
  GraphUtil(ObjMesh &mesh_graph, const std::vector<int> &vertex_index): edge_set_(new EdgeSet(mesh_graph)) {
    mesh_graph_ = &mesh_graph;
    vertex_index_ = vertex_index;
    std::vector<double> initialerVI(mesh_graph_->n_verts_, DBL_MAX);
    dist_.resize(vertex_index_.size(), initialerVI);

  }


  
  void FindNearest();
  void ColorizeNearestVertices();


public:
  std::unique_ptr<EdgeSet> edge_set_;
  std::vector<int> vertex_index_;
  ObjMesh * mesh_graph_;
  std::vector<std::vector<double>> dist_;
};