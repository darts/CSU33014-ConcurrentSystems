//
// CSU33014 Summer 2020 Additional Assignment
// Part B of a two-part assignment
//
// Please write your solution in this file

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "csu33014-annual-partB-person.h"

void find_reachable_recursive(struct person *current, int steps_remaining,
                              bool *reachable)
{
  // mark current root person as reachable
  reachable[person_get_index(current)] = true;
  // now deal with this person's acquaintances
  if (steps_remaining > 0)
  {
    int num_known = person_get_num_known(current);
    for (int i = 0; i < num_known; i++)
    {
      struct person *acquaintance = person_get_acquaintance(current, i);
      find_reachable_recursive(acquaintance, steps_remaining - 1, reachable);
    }
  }
}

// computes the number of people within k degrees of the start person
int number_within_k_degrees(struct person *start, int total_people, int k)
{
  bool *reachable;
  int count;

  // maintain a boolean flag for each person indicating if they are visited
  reachable = malloc(sizeof(bool) * total_people);
  for (int i = 0; i < total_people; i++)
  {
    reachable[i] = false;
  }

  // now search for all people who are reachable with k steps
  find_reachable_recursive(start, k, reachable);

  // all visited people are marked reachable, so count them
  count = 0;
  for (int i = 0; i < total_people; i++)
  {
    if (reachable[i] == true)
    {
      count++;
    }
  }
  return count;
}

// A more efficient version of the algorithm
// Uses a breadth-first-search approach
int find_reachable_bfs(struct person *start, int steps_remaining, int total_people)
{
  // maintain a boolean flag for each person indicating if they are visited
  bool *reachable = calloc(total_people, sizeof(bool));
  int count = 1;

  // two stacks for keeping track of people to visit in this iteration an the next
  struct person **a_arr = malloc(sizeof(struct person) * total_people);
  struct person **b_arr = malloc(sizeof(struct person) * total_people);
  bool popA = true;
  int num_in_next = 1;                       // how many to search in next iteration
  a_arr[0] = start;                          // add first node
  reachable[person_get_index(start)] = true; // mark first person as visited

  while (steps_remaining-- > 0 && num_in_next > 0)
  {
    //where to start writing to array
    int pushIndex = 0;

    //Which stack to read from and which to write to
    struct person **cur_pop = popA ? a_arr : b_arr;  //All of the people to search at this depth are on this stack
    struct person **cur_push = popA ? b_arr : a_arr; //All of the people to visit in the next iteration on this stack

    //go through all the people to visit at this depth
    for (int popIndex = 0; popIndex < num_in_next; popIndex++)
    {
      //get the number of people this person knows
      int num_known = person_get_num_known(cur_pop[popIndex]);

      //Go through everyone they know
      for (int i = 0; i < num_known; i++)
      {
        struct person *acquaintance = person_get_acquaintance(cur_pop[popIndex], i);
        //If this person has not yet been seen, add them to the list, otherwise ignore them
        if (reachable[person_get_index(acquaintance)] == false)
        {
          reachable[person_get_index(acquaintance)] = true;
          cur_push[pushIndex++] = acquaintance;
          count++;
        }
      }
    }

    //switch stacks
    popA = !popA;

    //set number of people to check in next iteration
    num_in_next = pushIndex;
  }
  //clean up and return
  free(a_arr);
  free(b_arr);
  free(reachable);
  return count;
}

// computes the number of people within k degrees of the start person;
// less repeated computation than the simple original version
int less_redundant_number_within_k_degrees(struct person *start,
                                           int total_people, int k)
{
  return find_reachable_bfs(start, k, total_people);
}
/*
Part B.1 Comment

Original Algorithm: Recursive Depth First Search with duplication
The algorithm in the original code is a recursive depth first search algorithm. It will recursively travel through every person/node in the graph until a desired depth is reached.
Due to this, it has the potential to be very inefficient in searching. The worst case running time of the function in Big O notation is O(n^k), where n is the number of people in
the graph and k is the desired depth. The scenario for this worst-case performance is a fully connected graph. If the graph is fully connected, at each stage there will be n nodes
investigated with duplication. That means that for a depth of 1: n, depth of 2: n*n, depth of 3: n*n*n --> O(n^k)

New algorithm: Iterative Breadth First Search without duplication
There were several design considerations when choosing an algorithm including general speed and the ability to be parallelised. The latter primarily resulted in the choice of using
an iterative algorithm over a recursive one. The reason for this is the extra complexity involved with parallelising a recursive algorithm. Not only this but for this problem, 
there was little benefit to using a recursive algorithm. It increased stack memory usage and introduced overhead. 
The algorithm follows a breadth first search approach, that is, rather than following a graph all the way down to the desired depth immediately, this searching algorithm will 
investigate all nodes at each depth. Take the following graph as an example.
     1
   /   \
  2     3
 / \   / \
4   5 6   7
In the above graph, a depth first search algorithm will traverse the nodes in the following order (starting at 1): 1, 2, 4, 5, 3, 6, 7. A breadth first search algorithm (again 
starting at 1) will traverse the nodes as: 1, 2, 3, 4, 5, 6, 7.
The latter approach allows for the program to be parallelised more easily. As at each stage there are more individual paths to follow. This can be seen if we show the order that 
nodes were visited into sections of each pass of the algorithm: BFS -> (1), (2,3), (4,5,6,7)     DFS -> (1 (2 (4)(5) 3 (6)(7))). This demonstrates the reduction of complexity that 
an iterative BFS algorithm provides. 
Actual performance improvements are brought about by tracking which drives have already been visited. By doing this we reduce the worst case running time of the function to O(n).
This is because each node can only be visited once. This prevents the issue seen in the original algorithm where each node would be repeatedly checked. The way this is implemented 
is through the use of an array of booleans, similar to how nodes were counted in the original. At each depth, the nodes that were previously added to the list are iterated through,
for each node and its child nodes are added to the list of nodes to search at the next depth only if they have not already been visited. Upon being added to the list, nodes are 
marked as visited, this prevents them from being added again resulting in the worst case running time of O(n) as in the worst case scenario all of the nodes are traversed.


A concession: this particular implementation has a slightly larger memory overhead than could be acheived due to the use of statically sized lists. This was a choice made as 
the number of people is quite small and thus the memory cost for creating the arrays is acceptable. Using dynamically sized lists could have resulted in significant overhead when
resizing the arrays. In the case of a graph with many more nodes, it is quite possible that the overhead in using dynamically sized lists would be worth the trade off in memory 
costs.
*/

// A more efficient version of the algorithm
// Uses a parallel breadth-first-search approach
int find_reachable_bfs_p(struct person *start, int steps_remaining, int total_people)
{
  // maintain a boolean flag for each person indicating if they are visited
  bool *reachable = calloc(total_people, sizeof(bool));
  int count = 1;

  // two stacks for keeping track of people to visit in this iteration an the next
  struct person **a_arr = malloc(sizeof(struct person) * total_people);
  struct person **b_arr = malloc(sizeof(struct person) * total_people);
  bool popA = true;
  int num_in_next = 1;                       // how many to search in next iteration
  a_arr[0] = start;                          // add first node
  reachable[person_get_index(start)] = true; // mark first person as visited

  while (steps_remaining-- > 0 && num_in_next > 0)
  {
    //where to start writing to array
    int pushIndex = 0;

    //Which stack to read from and which to write to
    struct person **cur_pop = popA ? a_arr : b_arr;  //All of the people to search at this depth are on this stack
    struct person **cur_push = popA ? b_arr : a_arr; //All of the people to visit in the next iteration on this stack

    //go through all the people to visit at this depth
    // the reduction part is important to keep the count safe, this was found after some painful debugging
    // The effect of 'reduction' in this case can be approximated by:
    /*
      count = 0;
      each_thread{
        local_count = 0;
        doBFS();
      }.onThreadClose{
        atomic_operation(count += local_count);
      }
    */
    #pragma omp parallel reduction(+:count) 
    for (int popIndex = 0; popIndex < num_in_next; popIndex++)
    {
      //get the number of people this person knows
      int num_known = person_get_num_known(cur_pop[popIndex]);

      //Go through everyone they know
      for (int i = 0; i < num_known; i++)
      {
        struct person *acquaintance = person_get_acquaintance(cur_pop[popIndex], i);
        //If this person has not yet been seen, add them to the list, otherwise ignore them
        if (reachable[person_get_index(acquaintance)] == false)
        {
          reachable[person_get_index(acquaintance)] = true;
          cur_push[pushIndex++] = acquaintance;
          count++;
        }
      }
    }

    //switch stacks
    popA = !popA;

    //set number of people to check in next iteration
    num_in_next = pushIndex;
  }
  //clean up and return
  free(a_arr);
  free(b_arr);
  free(reachable);
  return count;
}

// computes the number of people within k degrees of the start person;
// parallel version of the code
int parallel_number_within_k_degrees(struct person *start, int total_people,
                                     int k)
{
  return find_reachable_bfs_p(start, total_people, k);
}
