#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>
#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_errno.h>
#include <mpi.h>

struct barrier_request {
	struct fid		*sched_fid;
	struct fi_context	*commands;
	uint64_t		send_msg;
	uint64_t		recv_msg;
	void			*context;
	uint32_t		num_commands;
};

int prepare_barrier(struct fid_ep *ep, fi_addr_t *group,
		int myrank, int nranks, uint64_t tag,
		struct barrier_request *breq)
{
	int i, j, ret, dst, src, mask, nsteps = 0;
	struct fi_msg_tagged msg = {0};
	struct fi_sched *sched;
	struct iovec iov;

	/* count the number of steps */
	mask = 0x1;
	while (mask < nranks) {
		nsteps++;
		mask <<= 1;
	}

	sched = malloc(sizeof(struct fi_sched) * nsteps);
	if (!sched) {
		fprintf(stderr, "no memory\n");
		return -ENOMEM;
	}

	/* prepare the schedule */
	mask = 0x1;
	for (i = 0, j = 0, mask = 0x1; mask < nranks; i+=2, j++, mask <<= 1) {
		dst = (myrank + mask) % nranks;
		src = (myrank - mask + nranks) % nranks;

		/*
		mpi_errno = MPIC_Sendrecv(NULL, 0, MPI_BYTE, dst,
				MPIR_BARRIER_TAG, NULL, 0, MPI_BYTE,
				src, MPIR_BARRIER_TAG, comm_ptr,
				MPI_STATUS_IGNORE, errflag);
		*/
		iov.iov_base = &breq->send_msg;
		iov.iov_len  = sizeof(breq->send_msg);
		msg.msg_iov   = &iov;
		msg.iov_count = 1;
		msg.addr = group[dst];
		msg.tag  = tag;
		msg.context = &breq->commands[i];

		ret = fi_tsendmsg(ep, &msg, FI_SCHEDULE);
		if (ret) {
			fprintf(stderr, "fi_tsendmsg (%s)\n", fi_strerror(ret));
			return ret;
		}

		iov.iov_base = &breq->recv_msg;
		iov.iov_len  = sizeof(breq->recv_msg);
		msg.addr = group[src];
		msg.context = &breq->commands[i+1];
		ret = fi_trecvmsg(ep, &msg, FI_SCHEDULE);
		if (ret) {
			fprintf(stderr, "fi_trecvmsg (%s)\n", fi_strerror(ret));
			return ret;
		}

		/* two operations per phase */
		sched[j].ops = malloc(sizeof(struct fi_context *) * 2);
		if (!sched[j].ops) {
			fprintf(stderr, "no memory\n");
			return -ENOMEM;
		}

		sched[j].ops[0] = &breq->commands[i];
		sched[j].ops[1] = &breq->commands[i+1];
		sched[j].num_ops = 2;

		if (j+1 == nsteps) {
			sched[j].edges = NULL;
			sched[j].num_edges = 0;
		} else {
			sched[j].edges = malloc(sizeof(struct fi_sched *));
			if (!sched[j].edges) {
				fprintf(stderr, "no memory\n");
				return -ENOMEM;
			}
			sched[j].edges[0] = &sched[j+1];
			sched[j].num_edges = 1;
		}
	}

	ret = fi_sched_open(ep, &sched[0], &breq->sched_fid,
			0, breq->context);
	if (ret) {
		fprintf(stderr, "fi_sched_format (%s)\n", fi_strerror(ret));
		return ret;
	}

	for (i = 0; i < nsteps; i++) {
		free(sched[i].ops);
		free(sched[i].edges);
	}

	free(sched);

	MPI_Barrier(MPI_COMM_WORLD);

	return 0;
}

int start_barrier(struct barrier_request *breq)
{
	int ret;

	ret = fi_sched_start(breq->sched_fid);
	if (ret) {
		fprintf(stderr, "fi_sched_start (%s)\n", fi_strerror(ret));
		return ret;
	}

	return 0;
}

int wait_barrier(struct fid_cq *cq, struct barrier_request *breq)
{
	int ret;
	struct fi_cq_entry cqe = {0};

	do {
		ret = fi_cq_read(cq, &cqe, 1);
		if (ret > 0) {
			if (cqe.op_context != breq->context) {
				fprintf(stderr, "err comp context %p, expected %p\n",
						cqe.op_context, breq->context);
				return -1;
			} else {
				printf("Barrier complete %p\n", breq->context);
				break;
			}
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			fprintf(stderr, "fi_cq_read (%s)\n", fi_strerror(ret));
			return ret;
		}
	} while(1);

	return 0;
}

int main(int argc, char* argv[])
{
	int    i, j, ret, myrank, nranks;
	int    iterations = 1, num_out_barriers = 1;

	struct fi_info		*info, *hints;
	struct fid_fabric	*fabric;
	struct fid_domain	*domain;
	struct fid_cq		*cq;
	struct fid_av		*av;
	struct fid_ep		*ep;
	struct fi_cq_attr 	cq_attr = {
					.size 	= 128,
					.format = FI_CQ_FORMAT_CONTEXT,
				};
	struct fi_av_attr	av_attr = {
					.type   = FI_AV_MAP,
				};
	struct barrier_request *breq;

	char epname[128];
	size_t  epnamelen = sizeof(epname);
	char *allepnames;
	fi_addr_t *group;

	MPI_Init(&argc, &argv);

	if (argc > 1) {
		iterations = atoi(argv[1]);
	}
	if (argc > 2) {
		num_out_barriers = atoi(argv[2]);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	breq = malloc(sizeof(struct barrier_request) * num_out_barriers);
	if (!breq) {
		return -ENOMEM;
	}

	for(i = 0; i < num_out_barriers; i++) {
		breq[i].send_msg = myrank;
		breq[i].recv_msg = UINT64_MAX;
		breq[i].context = (void *) &breq[i];
	}

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->ep_attr->type	= FI_EP_RDM;
	hints->caps		= FI_TAGGED | FI_SCHEDULE;
	hints->domain_attr->data_progress = FI_PROGRESS_AUTO;

	ret = fi_getinfo(FI_VERSION(1, 3), NULL,
			NULL, 0, hints, &info);
	if (ret) {
		fprintf(stderr, "fi_getinfo (%s)\n", fi_strerror(ret));
	}

	ret = fi_fabric(info->fabric_attr, &fabric, NULL);
	if (ret) {
		fprintf(stderr, "fi_fabric (%s)\n", fi_strerror(ret));
		return ret;
	}
	if (!info) {
		fprintf(stderr, "no available providers\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_domain(fabric, info, &domain, NULL);
	if (ret) {
		fprintf(stderr, "fi_domain (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
	if (ret) {
		fprintf(stderr, "fi_cq_open (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		fprintf(stderr, "fi_av_open (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_endpoint(domain, info, &ep, NULL);
	if (ret) {
		fprintf(stderr, "fi_endpoint (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret) {
		fprintf(stderr, "fi_ep_bind (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
	if (ret) {
		fprintf(stderr, "fi_ep_bind (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_enable(ep);
	if (ret) {
		fprintf(stderr, "fi_enable (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_getname(&ep->fid, epname, &epnamelen);
	if (ret) {
		fprintf(stderr, "fi_getname (%s)\n", fi_strerror(ret));
		return ret;
	}

	allepnames = malloc(nranks * epnamelen);
	if (!allepnames) {
		fprintf(stderr, "no memory for epnames\n");
		return -ENOMEM;
	}

	group = malloc(nranks * sizeof(fi_addr_t));
	if (!group) {
		fprintf(stderr, "no memory for remote addresses\n");
		return -ENOMEM;
	}

	MPI_Allgather(epname, epnamelen, MPI_BYTE,
			allepnames, epnamelen, MPI_BYTE, MPI_COMM_WORLD);

	for(i = 0; i < nranks; i++) {
		void *addr;
		if (i == myrank)
			continue;
		ret = fi_av_insert(av, &allepnames[i*epnamelen], 1,
				&group[i], 0, NULL);
		if (ret != 1) {
			fprintf(stderr, "fi_av_insert (%s)\n", fi_strerror(ret));
			return ret;
		}
	}

	for(i = 0; i < num_out_barriers; i++) {
		ret = prepare_barrier(ep, group, myrank, nranks,
				(uint64_t) i, &breq[i]);
		if (ret)
			return ret;
	}

	for(i = 0; i < iterations; i++) {
		for(j = 0; j < num_out_barriers; j++) {
			ret = start_barrier(&breq[j]);
			if (ret)
				return ret;
		}
		for(j = 0; j < num_out_barriers; j++) {
			ret = wait_barrier(cq, &breq[j]);
			if (ret)
				return ret;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	for(i = 0; i < num_out_barriers; i++) {
		fi_close(breq[i].sched_fid);
		free(breq[i].commands);
	}

	fi_close(&ep->fid);
	fi_close(&av->fid);
	fi_close(&cq->fid);
	fi_close(&domain->fid);
	fi_close(&fabric->fid);

	free(group);
	free(allepnames);

	MPI_Finalize();
	return ret;
}
