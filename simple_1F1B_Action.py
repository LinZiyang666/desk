from pipelining_source_code.schedules import _Action, _ComputationType
from typing import Dict, List, Tuple, Optional

def generate_1f1b_pipeline_actions(num_stages: int, num_microbatches: int, upstream = None):
    actions_per_rank = {}

    for stage_idx in range(num_stages):
        rank = stage_idx  # stage == rank
        actions = []
        recv_actions = []
        local_id = 0

        warmup_chunks = min(num_microbatches, num_stages - stage_idx)

        fwd_mb = 0  
        bwd_mb = 0  
        # [stage],[rank],[id],[action type],[microbatch],[dest_rank],[upstream],[dependency]
        def make_action(stage, rank, type_, mb, dest):
            nonlocal local_id
            a = _Action(stage, rank, local_id, type_, mb, dest, upstream, None)
            local_id += 1
            return a

        # Warmup
        for _ in range(warmup_chunks):
            if stage_idx > 0:
                actions.append(make_action(stage_idx, rank, _ComputationType.RECV_F, fwd_mb, rank - 1))
            actions.append(make_action(stage_idx, rank, _ComputationType.FORWARD, fwd_mb, None))
            if stage_idx < num_stages - 1:
                actions.append(make_action(stage_idx, rank, _ComputationType.SEND_F, fwd_mb, rank + 1))
            fwd_mb += 1

        # 1B1F
        while fwd_mb < num_microbatches:
            if stage_idx < num_stages - 1:
                actions.append(make_action(stage_idx, rank, _ComputationType.RECV_B, bwd_mb, rank + 1))
            actions.append(make_action(stage_idx, rank, _ComputationType.FULL_BACKWARD, bwd_mb, None))
            if stage_idx > 0:
                actions.append(make_action(stage_idx, rank, _ComputationType.SEND_B, bwd_mb, rank - 1))
            bwd_mb += 1

            if stage_idx > 0:
                actions.append(make_action(stage_idx, rank, _ComputationType.RECV_F, fwd_mb, rank - 1))
            actions.append(make_action(stage_idx, rank, _ComputationType.FORWARD, fwd_mb, None))
            if stage_idx < num_stages - 1:
                actions.append(make_action(stage_idx, rank, _ComputationType.SEND_F, fwd_mb, rank + 1))
            fwd_mb += 1

        # Cooldown
        while bwd_mb < num_microbatches:
            if stage_idx < num_stages - 1:
                actions.append(make_action(stage_idx, rank, _ComputationType.RECV_B, bwd_mb, rank + 1))
            actions.append(make_action(stage_idx, rank, _ComputationType.FULL_BACKWARD, bwd_mb, None))
            if stage_idx > 0:
                actions.append(make_action(stage_idx, rank, _ComputationType.SEND_B, bwd_mb, rank - 1))
            bwd_mb += 1

        actions_per_rank[rank] = actions

    return actions_per_rank


def generate_1f1b_pipeline_actions_pro(num_stages: int,total_samples: int, num_microbatches: int,group_info, batch_info, upstream = None):

    per_mbatch_size = int(total_samples/num_microbatches)
    #each time, we should send/calculate a bunch of samples...

    actions_per_rank = {}

    print(batch_info)

    for stage_idx in range(num_stages):
        for cur_rank_batch_pair in batch_info[stage_idx]:
            cur_rank = cur_rank_batch_pair[0]  # get the current focused rank index...
            sample_chunk = cur_rank_batch_pair[1] # this records relative idx within one microbatches only...
            ### now, we get the sample chunk this rank should focus on...

            actions = []
            recv_actions = []
            local_id_comp = total_samples
            local_id_comm = 0

            warmup_chunks = min(num_microbatches, (num_stages - stage_idx)*2-1)  #the stride is 2 per...
            
            fwd_mb_index = 0  
            bwd_mb_index = 0  

            # [stage],[rank],[id],[action type],[microbatch],[dest_rank],[upstream],[dependency]
            def make_action(stage, rank, type_, mb, dest):
                nonlocal local_id_comp
                nonlocal local_id_comm
                if type_ == _ComputationType.RECV_F or type_ == _ComputationType.RECV_B:
                    a = _Action(stage, rank, local_id_comm, type_, mb, dest, upstream, None)
                    local_id_comm += 1
                else:
                    a = _Action(stage, rank, local_id_comp, type_, mb, dest, upstream, None)
                    local_id_comp += 1                    
                return a
            

            next_group_info = batch_info[stage_idx+1] if stage_idx+1<num_stages else None
            prev_group_info = batch_info[stage_idx-1] if stage_idx-1>=0 else None

            def make_comm_action(mb_index, type_, group_info):
                nonlocal sample_chunk, stage_idx, per_mbatch_size
                bias = mb_index*per_mbatch_size
                gathered_action = []
                if group_info == None: return gathered_action
                min_index = sample_chunk[0]
                max_index = sample_chunk[-1]
                for receiver_candidate_pair in group_info:
                    candidate_index = receiver_candidate_pair[0]
                    candidate_samples_chunk =  receiver_candidate_pair[1]
                    if min_index<=candidate_samples_chunk[-1]:
                        floor = min(candidate_samples_chunk[-1], max_index)
                        for idx_of_sample in range(min_index+bias, floor+1+bias):
                            gathered_action.append(make_action(stage_idx, cur_rank, type_,
                                                 idx_of_sample, candidate_index))
                        min_index = candidate_samples_chunk[-1]+1
                return gathered_action
            
            def make_comp_action(mb_index, type_):
                nonlocal sample_chunk, stage_idx, per_mbatch_size, cur_rank
                bias = mb_index*per_mbatch_size
                return [make_action(stage_idx, cur_rank, type_, tuple([x + bias for x in sample_chunk]) , None)]


            #add all necessary receives at the beginning: 
            ##jin: important!! here we test and assume that the communication is asyncronize...
            #so, we can do like this...


            #for i in range(num_microbatches):
            #    if stage_idx>0:
            #        actions.extend(make_comm_action(i, _ComputationType.RECV_F, prev_group_info))
            #for i in range(num_microbatches):
            #    if stage_idx< num_stages - 1:
            #        actions.extend(make_comm_action(i, _ComputationType.RECV_B, next_group_info))


            # Warmup
            for _ in range(warmup_chunks):
                #if stage_idx > 0:
                    #actions.append(make_action(stage_idx, rank, _ComputationType.RECV_F, fwd_mb, rank - 1))
                recv_actions.extend(make_comm_action(fwd_mb_index, _ComputationType.RECV_F, prev_group_info))
                actions.extend(make_comp_action(fwd_mb_index, _ComputationType.FORWARD))
                if stage_idx < num_stages - 1:
                    actions.extend(make_comm_action(fwd_mb_index, _ComputationType.SEND_F, next_group_info))
                    #actions.extend(make_action(stage_idx, rank, _ComputationType.SEND_F, fwd_mb, rank + 1))
                fwd_mb_index += 1

            # 1B1F
            while fwd_mb_index < num_microbatches:
                #if stage_idx < num_stages - 1:
                    #actions.append(make_action(stage_idx, rank, _ComputationType.RECV_B, bwd_mb, rank + 1))
                #actions.append(make_action(stage_idx, rank, _ComputationType.FULL_BACKWARD, bwd_mb, None))
                recv_actions.extend(make_comm_action(bwd_mb_index, _ComputationType.RECV_B, next_group_info))
                actions.extend(make_comp_action(bwd_mb_index, _ComputationType.FULL_BACKWARD))
                if stage_idx > 0:
                    #actions.append(make_action(stage_idx, rank, _ComputationType.SEND_B, bwd_mb, rank - 1))
                    actions.extend(make_comm_action(bwd_mb_index, _ComputationType.SEND_B, prev_group_info))
                bwd_mb_index += 1

                #if stage_idx > 0:
                    #actions.append(make_action(stage_idx, rank, _ComputationType.RECV_F, fwd_mb, rank - 1))
                #actions.append(make_action(stage_idx, rank, _ComputationType.FORWARD, fwd_mb, None))
                recv_actions.extend(make_comm_action(fwd_mb_index, _ComputationType.RECV_F, prev_group_info))
                actions.extend(make_comp_action(fwd_mb_index, _ComputationType.FORWARD))
                if stage_idx < num_stages - 1:
                    #actions.append(make_action(stage_idx, rank, _ComputationType.SEND_F, fwd_mb, rank + 1))
                    actions.extend(make_comm_action(fwd_mb_index, _ComputationType.SEND_F, next_group_info))
                fwd_mb_index += 1

            # Cooldown
            while bwd_mb_index < num_microbatches:
                #if stage_idx < num_stages - 1:
                    #actions.append(make_action(stage_idx, rank, _ComputationType.RECV_B, bwd_mb, rank + 1))
                #actions.append(make_action(stage_idx, rank, _ComputationType.FULL_BACKWARD, bwd_mb, None))
                recv_actions.extend(make_comm_action(bwd_mb_index, _ComputationType.RECV_B, next_group_info))
                actions.extend(make_comp_action(bwd_mb_index, _ComputationType.FULL_BACKWARD))
                if stage_idx > 0:
                    #actions.append(make_action(stage_idx, rank, _ComputationType.SEND_B, bwd_mb, rank - 1))
                    actions.extend(make_comm_action(bwd_mb_index, _ComputationType.SEND_B, prev_group_info))
                bwd_mb_index += 1

            if len(batch_info[stage_idx])>1:
                actions.append(_Action(stage_idx, cur_rank, local_id_comp, _ComputationType.ALL_REDUCE, None, upstream = 10000))
            actions_per_rank[cur_rank] = recv_actions + actions

    return actions_per_rank




def print_pipeline_actions(num_stages, num_microbatches):
    actions = generate_1f1b_pipeline_actions(num_stages, num_microbatches)
    
    print("def create_pipeline_actions():")
    
    for rank in range(num_stages):
        print(f"\n    # Rank {rank} (Stage {rank}) 的操作序列")
        print(f"    rank{rank}_actions = [")
        for action in actions[rank]:
            print(f"        {action},")
        print("    ]")
    
    print("\n    return {" + ", ".join([f"{rank}: rank{rank}_actions" for rank in range(num_stages)]) + "}")
    
    return actions


if __name__ == "__main__":
    print_pipeline_actions(3, 8)
    
    
    
    
def add_comm_dependencies(communication_order, actions_per_rank):
    """
    communication_order: List[Tuple[int, int, int]]
        Each tuple is (rank, microbatch_index, send_flag) where send_flag: 1->F, 0->B.
        We interpret every tuple as a SEND. For each tuple after the first, its SEND will
        depend on the RECV corresponding to the previous tuple.

    actions_per_rank: Dict[int, List[_Action]]
        Mutated in-place. For each current SEND, we add a dependency on the RECV that
        matches the previous tuple's (rank, microbatch, F/B).
    """
    # Local helpers (kept inside so the top-level API is a single function)
    def mb_matches(action_mb, mb_val: int) -> bool:
        # action_mb could be int or tuple[int, ...]
        return (mb_val in action_mb) if isinstance(action_mb, tuple) else (action_mb == mb_val)

    # Resolve enum values without importing here
    # Expecting names on your side: SEND_F, SEND_B, RECV_F, RECV_B
    def send_flag_to_ct(flag):
        return getattr(_ComputationType, "SEND_F" if flag == 1 else "SEND_B")

    def recv_flag_to_ct(flag):
        return getattr(_ComputationType, "RECV_F" if flag == 1 else "RECV_B")

    prev = None
    for cur in communication_order:
        if prev is None:
            prev = cur
            continue

        send_rank, send_mb, send_flag = cur
        prev_rank, prev_mb, prev_flag = prev

        # 1) Find the current SEND action on send_rank that matches (send_mb, send_flag)
        send_ct = send_flag_to_ct(send_flag)
        send_action = None
        for a in actions_per_rank.get(send_rank, ()):
            if a.computation_type == send_ct and mb_matches(a.microbatch_index, send_mb):
                send_action = a
                break
        if send_action is None:
            raise RuntimeError(f"SEND not found for {cur}")

        # 2) Find the RECV action that pairs with the *previous* SEND tuple
        #    We look for RECV_* whose dest_rank == prev_rank and mb matches prev_mb.
        recv_ct = recv_flag_to_ct(prev_flag)
        recv_action = None
        for r, acts in actions_per_rank.items():
            for a in acts:
                if (a.computation_type == recv_ct
                    and a.dest_rank == prev_rank
                    and mb_matches(a.microbatch_index, prev_mb)):
                    recv_action = a
                    break
            if recv_action is not None:
                break
        if recv_action is None:
            raise RuntimeError(f"RECV not found for previous {prev}")

        # 3) Wire dependency: current SEND depends on that RECV
        #    dependency: Dict[int, Tuple[int, ...]] where key = rank of depended action
        dep: Dict[int, Tuple[int, ...]] = send_action.dependency or {}
        existed = list(dep.get(recv_action.rank, ()))
        if recv_action.id not in existed:
            existed.append(recv_action.id)
        dep[recv_action.rank] = tuple(existed)
        send_action.set_dependency(dep)

        prev = cur

