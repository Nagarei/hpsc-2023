#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 10000;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Body ibody[N/size], jbody[2][N/size];
  srand48(rank);
  for(int i=0; i<N/size; i++) {
    ibody[i].x = jbody[0][i].x = drand48();
    ibody[i].y = jbody[0][i].y = drand48();
    ibody[i].m = jbody[0][i].m = drand48();
    ibody[i].fx = jbody[0][i].fx = ibody[i].fy = jbody[0][i].fy = 0;
  }
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");
  MPI_Win win;
  MPI_Win_create(jbody[0], (N/size) * 2 * sizeof(Body), sizeof(Body), info, MPI_COMM_WORLD, &win);
  for(int irank=0; irank<size; irank++) {
    const int datai = irank&1;
    const int buffi = datai^1;
    MPI_Win_fence(0,win);
    if (irank+1 < size) {
      //MPI_Send(jbody, N/size, MPI_BODY, send_to, 0, MPI_COMM_WORLD);
      MPI_Put(jbody[datai], N/size, MPI_BODY, send_to, (N/size)*buffi, N/size, MPI_BODY, win);
      //MPI_Recv(jbody, N/size, MPI_BODY, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for(int i=0; i<N/size; i++) {
      for(int j=0; j<N/size; j++) {
        double rx = ibody[i].x - jbody[datai][j].x;
        double ry = ibody[i].y - jbody[datai][j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[datai][j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[datai][j].m / (r * r * r);
        }
      }
    }
  }
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        printf("%d %g %g\n",i+rank*N/size,ibody[i].fx,ibody[i].fy);
      }
      fflush(stdout);
    }
  }
  MPI_Finalize();
}
