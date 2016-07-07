#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>
#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_errno.h>
#include <mpi.h>

#define SEND 1
#define RECV 2
#define SEND_ATOMIC 3

/* Integer reduction using binary trees */

struct reduce_request {
	struct fid_sched	*sched_fid;
	struct fi_sched_ops	*sched_ops;
	struct fid_cntr		*cntr;
	uint32_t		send_msg;
	uint32_t		recv_msg;
	void			*context;
};

int prepare_cmd(struct fid_ep *ep, void *buf, size_t len, fi_addr_t addr,
		uint64_t tag, void *context, int cmd_type)
{
	int ret;
	struct iovec iov;
	struct fi_ioc ioc;
	struct fi_msg_tagged msg = {0};
	struct fi_msg_atomic atomic_msg = {0};

	switch(cmd_type) {
	case SEND:
	case RECV:
		iov.iov_base = buf;
		iov.iov_len  = len;
		msg.msg_iov   = &iov;
		msg.iov_count = 1;
		msg.addr = addr;
		msg.tag  = tag;
		msg.context = context;
		if (cmd_type == SEND) {
			ret = fi_tsendmsg(ep, &msg, FI_SCHEDULE);
			if (ret) {
				fprintf(stderr, "fi_tsendmsg (%s)\n", fi_strerror(ret));
				return ret;
			}
		} else {
			ret = fi_trecvmsg(ep, &msg, FI_SCHEDULE);
			if (ret) {
				fprintf(stderr, "fi_trecvmsg (%s)\n", fi_strerror(ret));
				return ret;
			}
		}
		break;
	case SEND_ATOMIC:
		ioc.addr = (void *) buf;
		ioc.count = len;

		atomic_msg.msg_iov = &ioc;
		atomic_msg.desc = NULL;
		atomic_msg.iov_count = 1;
		atomic_msg.addr = addr;
		atomic_msg.tag = tag;
		atomic_msg.context = context;
		atomic_msg.data = 0;
		atomic_msg.tag = tag;
		atomic_msg.datatype = FI_UINT32;
		atomic_msg.op = FI_SUM;

		ret = fi_tsend_atomicmsg(ep, &atomic_msg, FI_SCHEDULE);
		if (ret) {
			fprintf(stderr, "fi_tsend_atomicmsg (%s)\n", fi_strerror(ret));
			return ret;
		}
		break;
	default:
		return -1;
	}

	return 0;
}

int child_op(struct fid_ep *ep, struct fi_sched_ops *sched_ops, uint32_t *msg,
		fi_addr_t *group, int left_child, uint64_t tag,
		int num_children, int op)
{
	int i, ret;

	sched_ops->ops = malloc(sizeof(struct fi_context) * num_children);
	if (!sched_ops->ops)
		return -FI_ENOMEM;

	sched_ops->num_ops = num_children;

	for(i=0; i<num_children; i++) {
		ret = prepare_cmd(ep, msg, sizeof(uint32_t),
				group[left_child+i],
				tag, &sched_ops->ops[i], op);
		if (ret)
			return ret;
	}

	return 0;
}

int parent_op(struct fid_ep *ep, struct fi_sched_ops *sched_ops, uint32_t *msg,
		uint64_t tag, fi_addr_t parent_addr, int op)
{
	int ret;
	size_t len;

	sched_ops->ops = malloc(sizeof(struct fi_context));
	if (!sched_ops->ops)
		return -FI_ENOMEM;

	sched_ops->num_ops = 1;

	if (op == SEND_ATOMIC)
		len = 1;
	else
		len = sizeof(uint32_t);

	ret = prepare_cmd(ep, msg, len,
			parent_addr, tag,
			&sched_ops->ops[0], op);
	if (ret)
		return ret;

	return 0;
}

int create_schedule_phase_array(struct fi_sched_ops **sched_ops, int steps)
{
	int i;
	struct fi_sched_ops *ops;

	ops = malloc(sizeof(struct fi_sched_ops) * steps);
	if (!ops)
		return -FI_ENOMEM;

	for(i=0; i<steps; i++) {
		ops[i].num_edges = 1;
		if (i == steps-1)
			ops[i].edges = NULL;
		else
			ops[i].edges = &ops[i+1];
	}

	*sched_ops = ops;

	return 0;
}

int init_reduce(struct fid_domain *domain, struct fid_ep *ep, fi_addr_t *group,
		int myrank, int nranks, uint64_t tag,
		struct reduce_request *rreq)
{
	int root=0, intermediate=0, leaf=0, ret,
	    left_child, right_child, parent, num_children=0;
	struct fi_cntr_attr	cntr_attr = {
					.events = FI_CNTR_EVENTS_COMP,
				};

	left_child = 2*(myrank+1)-1;
	right_child = left_child+1;

	if (myrank == 0)
		root = 1;
	else if (left_child > nranks)
		leaf = 1;
	else
		intermediate = 1;

	if (!leaf) {
		if (left_child == (nranks-1))
			num_children = 1;
		else
			num_children = 2;
	}

	fprintf(stderr, "[%d] left_child %d right_child %d nranks %d num_children %d"
			"(root %d intermediate %d leaf %d)\n",
			myrank, left_child, right_child, nranks, num_children,
			root, intermediate, leaf);

	parent = myrank/2;

	/* initialize receive buffer with my contribution */
	rreq->recv_msg = rreq->send_msg;

	if (root) {
		/* step 1: wait for children
		 * step 2: broadcast down to children */

		ret = create_schedule_phase_array(&rreq->sched_ops, 2);
		if (ret)
			return ret;

		rreq->sched_ops = malloc(sizeof(struct fi_sched_ops) * 2);
		if (!rreq->sched_ops)
			return -FI_ENOMEM;

		ret = child_op(ep, &rreq->sched_ops[0], &rreq->recv_msg,
				group, left_child, tag, num_children, RECV);
		if (ret)
			return ret;

		ret = child_op(ep, &rreq->sched_ops[1], &rreq->recv_msg,
				group, left_child, tag, num_children, SEND);
		if (ret)
			return ret;

	} else if (intermediate) {
		/* step 1: wait for children
		 * step 2: send up to parent
		 * step 3: wait for parent
		 * step 4: send down to children */

		ret = create_schedule_phase_array(&rreq->sched_ops, 4);
		if (ret)
			return ret;

		ret = child_op(ep, &rreq->sched_ops[0], &rreq->recv_msg,
				group, left_child, tag, num_children, RECV);
		if (ret)
			return ret;

		ret = parent_op(ep, &rreq->sched_ops[1],
				&rreq->recv_msg, tag, group[parent],
				SEND_ATOMIC);
		if (ret)
			return ret;

		ret = parent_op(ep, &rreq->sched_ops[2],
				&rreq->recv_msg, tag, group[parent],
				RECV);

		ret = child_op(ep, &rreq->sched_ops[3], &rreq->recv_msg,
				group, left_child, tag, num_children, SEND);

		if (ret)
			return ret;
	} else {
		/* step 1: send to parent
		 * step 2: wait for parent */
		ret = create_schedule_phase_array(&rreq->sched_ops, 2);
		if (ret)
			return ret;

		ret = parent_op(ep, &rreq->sched_ops[0], &rreq->recv_msg, tag,
				group[parent], SEND_ATOMIC);
		if (ret)
			return ret;

		ret = parent_op(ep, &rreq->sched_ops[1], &rreq->recv_msg, tag,
				group[parent], RECV);
		if (ret)
			return ret;
	}

	ret = fi_cntr_open(domain, &cntr_attr, &rreq->cntr, NULL);
	if (ret) {
		fprintf(stderr, "fi_cntr_open (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_sched_open(ep, &rreq->sched_fid, rreq->context);
	if (ret) {
		fprintf(stderr, "fi_sched_open (%s)\n", fi_strerror(ret));
		return ret;
	}

	ret = fi_sched_bind(rreq->sched_fid, &rreq->cntr->fid, 0);
	if (ret) {
		fprintf(stderr, "fi_sched_bind (%s)\n", fi_strerror(ret));
		return ret;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	return 0;
}

int finalize_reduce(struct reduce_request *rreq,
		int myrank, int nranks)
{
	int i, n, leaf, left_child;

	left_child = 2*(myrank+1)-1;

	if (myrank == 0)
		n = 2;
	else if (left_child > nranks)
		n = 4;
	else
		n = 2;

	for(i=0; i=n; i++)
		free(rreq->sched_ops[i].ops);

	free(rreq->sched_ops);

	fi_close(&rreq->cntr->fid);
	fi_close(&rreq->sched_fid->fid);
	return 0;
}

int start_reduce(struct reduce_request *rreq)
{
	int ret;

	ret = fi_sched_run(rreq->sched_fid);
	if (ret) {
		fprintf(stderr, "fi_sched_start (%s)\n", fi_strerror(ret));
		return ret;
	}

	return 0;
}

int wait_reduce(struct reduce_request *rreq, uint64_t threshold)
{
	int ret;

	ret = fi_cntr_wait(rreq->cntr, threshold, -1);
	if (ret < 0) {
		fprintf(stderr, "fi_cntr_wait (%s) error value %lu\n",
				fi_strerror(ret), fi_cntr_readerr(rreq->cntr));
		return ret;
	}

	return 0;
}

int check_reduce(uint32_t result, int nranks, int reduce_instance)
{
	float sum = (nranks/2.0)*(reduce_instance*2+nranks);

	if (result != (uint32_t) sum) {
		fprintf(stderr, "expected %u, but got %u\n",
				sum, result);
		return -FI_EOTHER;
	}
	return 0;
}

int main(int argc, char* argv[])
{
	int    i, j, ret, myrank, nranks;
	int    iterations = 1, num_reductions = 1;
	size_t count = 1;

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
	struct reduce_request *rreq;

	char epname[128];
	size_t  epnamelen = sizeof(epname);
	char *allepnames;
	fi_addr_t *group;

	MPI_Init(&argc, &argv);

	if (argc > 1) {
		iterations = atoi(argv[1]);
	}
	if (argc > 2) {
		num_reductions = atoi(argv[2]);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	rreq = malloc(sizeof(struct reduce_request) * num_reductions);
	if (!rreq) {
		return -ENOMEM;
	}

	for(i = 0; i < num_reductions; i++) {
		rreq[i].send_msg = myrank+i;
		rreq[i].recv_msg = 0;
		rreq[i].context = (void *) &rreq[i];
	}

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->ep_attr->type	= FI_EP_RDM;
	hints->caps		= FI_TAGGED | FI_ATOMIC | FI_SCHEDULE;
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

	ret = fi_tsend_atomicvalid(ep, FI_UINT32, FI_SUM, &count);
	if (ret) {
		fprintf(stderr, "fi_tsend_atomic not available\n");
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

	for(i = 0; i < num_reductions; i++) {
		ret = init_reduce(domain, ep, group, myrank, nranks,
				(uint64_t) i, &rreq[i]);
		if (ret)
			return ret;
	}

	for(i = 0; i < iterations; i++) {
		for(j = 0; j < num_reductions; j++) {
			ret = start_reduce(&rreq[j]);
			if (ret)
				return ret;
		}
		for(j = 0; j < num_reductions; j++) {
			ret = wait_reduce(&rreq[j], i+1);
			if (ret)
				return ret;

			ret = check_reduce(rreq[j].recv_msg,
					nranks, j);
			if (ret)
				return ret;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	for(i = 0; i < num_reductions; i++) {
		finalize_reduce(&rreq[i], myrank, nranks);
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
