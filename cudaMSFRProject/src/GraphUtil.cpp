/*!
 * \file GraphUtil.cpp
 * \date 2018/10/07 17:46
 *
 * \author sireer
 * Contact: sireerw@gmail.com
 *
 * \brief 
 *
 * TODO: long description
 *
 * \note
*/
#include "GraphUtil.h"
#include <iostream>
#include <fstream>
#include <queue>
class VD
{
public:
  VD(const int v_, const float d_):v(v_), d(d_){}
  bool operator<(const VD b) const
  {
    return d > b.d;
  }
public:
  int v;
  float d;
};

void GraphUtil::FindNearest() {
  //{
  //  std::deque<int> vQ;
  //  for (int i = 0; i < vertex_index_.size(); i++) {

  //    auto &dist = dist_[i];
  //    std::vector<bool> isInQ(mesh_graph_->n_verts_, false);

  //    vQ.push_back(vertex_index_[i]);
  //    isInQ[vertex_index_[i]] = true;
  //    dist[vertex_index_[i]] = 0;
  //    while (vQ.size() > 0) {
  //      auto head = vQ[0];
  //      for (auto i = edge_set_->begin(head); i != edge_set_->end(head); ++i) {
  //        if (dist[i->end_] > dist[head] + i->dist) {
  //          dist[i->end_] = dist[head] + i->dist;
  //          if (!isInQ[i->end_]) {
  //            vQ.push_back(i->end_);
  //            isInQ[i->end_] = true;
  //          }
  //        }
  //      }
  //      vQ.pop_front();
  //      isInQ[head] = false;
  //    }
  //  }
  //}
  {
    std::priority_queue<VD> vQ;
    for (int i = 0; i < vertex_index_.size(); i++)
    {
      auto &dist = dist_[i];
      std::vector<bool> isFind(mesh_graph_->n_verts_, false);
      vQ.push(VD(vertex_index_[i], 0.0));
      dist[vertex_index_[i]] = 0.0;
      while (vQ.size() > 0) {
        if (!isFind[vQ.top().v])
        {
          int head = vQ.top().v;
          isFind[head] = true;
          for (auto i = edge_set_->begin(head); i != edge_set_->end(head); ++i) {
            if (!isFind[i->end_] && dist[i->end_] > dist[head] + i->dist) {
              dist[i->end_] = dist[head] + i->dist;
              vQ.push(VD(i->end_, dist[i->end_]));
            }
          }
        }
        vQ.pop();
      }
    }
  }

  
  for (int i = 0; i < mesh_graph_->n_verts_; ++i)
  {
    mesh_graph_->geoposition_(i, 0) = dist_[0][i];
    mesh_graph_->geoposition_(i, 1) = dist_[1][i];
    mesh_graph_->geoposition_(i, 2) = dist_[2][i];
  }
}

void GraphUtil::ColorizeNearestVertices() {
  if (vertex_index_.size() == 3) {
    for (int i = 0; i < vertex_index_.size(); i++) {
      auto &dist = dist_[i];
      auto max = *std::max_element(dist.begin(), dist.end());
      for (int j = 0; j < mesh_graph_->n_verts_; j++) {
        mesh_graph_->color_(j, i) = dist[j] / max*0.7 + 0.3;
      }
    }
  }
}



EdgeSet::EdgeSet(const ObjMesh &obj){
  vertices_num_ = obj.n_verts_;

  {
    std::vector<Edge> edge_set_temp;
    for (int i = 0; i < obj.n_tri_; i++) {
      auto id0 = obj.tri_list_(i, 0);
      auto id1 = obj.tri_list_(i, 1);
      auto id2 = obj.tri_list_(i, 2);
      edge_set_temp.push_back(Edge(id0, id1));
      edge_set_temp.push_back(Edge(id1, id0));
      edge_set_temp.push_back(Edge(id1, id2));
      edge_set_temp.push_back(Edge(id2, id1));
      edge_set_temp.push_back(Edge(id2, id0));
      edge_set_temp.push_back(Edge(id0, id2));
    }

    std::sort(edge_set_temp.begin(), edge_set_temp.end(),
      [&](const Edge& x, const Edge&y) {
      if (x.start_ < y.start_)
        return true;
      else if (x.start_ == y.start_)
        return x.end_ < y.end_;
      else
        return false;
    });

    // remove repeated edges
    edge_set_.push_back(edge_set_temp[0]);
    for (auto i = edge_set_temp.begin(); i != edge_set_temp.end() - 1; ++i) {
      auto next = i + 1;
      if (i->start_ != next->start_ || i->end_ != next->end_) {
        edge_set_.push_back(*next);
      }
    }
  }

  //update look up table
  edges_num_ = edge_set_.size();
  head_.resize(vertices_num_);
  end_.resize(vertices_num_);
  head_[0] = edge_set_.begin();
  for (auto i = edge_set_.begin() + 1, pre = edge_set_.begin(); i != edge_set_.end(); pre=i, ++i) {
    if (i->start_ != pre->start_) {
      end_[pre->start_] = i;
      head_[i->start_] = i;
    }
  }
  end_[end_.size() - 1] = edge_set_.end();

  //compute dist

  for (auto &i : edge_set_) {
    auto start = obj.position_.block(i.start_, 0, 1, 3);
    auto end = obj.position_.block(i.end_, 0, 1, 3);
    i.dist = (start - end).norm();
  }
}
