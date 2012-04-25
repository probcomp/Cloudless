#!python
import datetime,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
reload(ds)
import DPMB_helper_functions as hf
reload(hf)
##
import pdb


class DPMB():
    def __init__(self,inf_seed,state,infer_alpha,infer_beta):
        nr.seed(int(np.clip(inf_seed,0,np.inf))) ##who's random seed is used where?  And does it even matter (consider true inf_seed to be f(inf_seed,gen_seed))?
        self.state = state
        self.infer_alpha = infer_alpha
        self.infer_beta = infer_beta
        ##
        self.transition_z_count = 0
    
    def transition_alpha_discrete_gibbs(self):
        start_dt = datetime.datetime.now()
        ##
        logp_list,lnPdf,grid = hf.calc_alpha_conditional(self.state)
        alpha_idx = hf.renormalize_and_sample(logp_list,self.state.verbose)
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,grid[alpha_idx])
        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["alpha"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["alpha"] = (datetime.datetime.now()-start_dt).seconds()
        self.state.timing["run_sum"] += self.state.timing["alpha"]

    def transition_beta_discrete_gibbs(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        ##
        for col_idx in range(self.state.num_cols):
            logp_list, lnPdf, grid = hf.calc_beta_conditional(self.state,col_idx)
            beta_idx = hf.renormalize_and_sample(logp_list)
            self.state.removeBetaD(lnPdf,col_idx)
            self.state.setBetaD(lnPdf,col_idx,grid[beta_idx])

            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if time_seatbelt is not None and self.state.timing["run_sum"] + delta_t > time_seatbelt:
                break 

        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).seconds()
        self.state.timing["run_sum"] += self.state.timing["betas"]

    def transition_alpha(self):
        if self.state.verbose:
            print "PRE transition_alpha score: ",self.state.score
        if self.infer_alpha:
            self.transition_alpha_discrete_gibbs()
        elif self.infer_alpha:
            self.state.timing["alpha"] = 0 ##ensure last value not carried forward

    def transition_beta(self,time_seatbelt=None):
        if self.state.verbose:
            print "PRE transition_beta score: ",self.state.score
        if self.infer_beta:
            self.transition_beta_discrete_gibbs(time_seatbelt)
        elif self.infer_beta:
            self.state.timing["betas"] = 0 ##ensure last value not carried forward
            
    def transition_z(self,time_seatbelt=None):
        self.transition_z_count += 1
        if self.state.verbose:
            print "PRE transition_z score: ",self.state.score
        start_dt = datetime.datetime.now()
        ##
        # for each vector
        for vector in nr.permutation(self.state.get_all_vectors()):
            
            # deassign it
            vector.cluster.deassign_vector(vector)
            
            # calculate the conditional
            score_vec = hf.calculate_cluster_conditional(self.state,vector)

            # sample an assignment
            draw = hf.renormalize_and_sample(score_vec)

            cluster = None
            if draw == len(self.state.cluster_list):
                cluster = self.state.generate_cluster_assignment(force_new = True)
            else:
                cluster = self.state.cluster_list[draw]

            # assign it
            cluster.assign_vector(vector)

            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if time_seatbelt is not None and self.state.timing["run_sum"] + delta_t > time_seatbelt:
                break  ## let logic below proceed

        # debug print out states:
        print " --- " + str(self.state.getZIndices())
        print "     " + str([cluster.count() for cluster in self.state.cluster_list])
        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["zs"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["zs"] = (datetime.datetime.now()-start_dt).seconds()
        self.state.timing["run_sum"] += self.state.timing["zs"]

    def transition_x():
        # regenerate new vector values, preserving the exact same clustering
        # create a new state, where you force init_z to be the current markov_chain, but you don't pass in init_x
        # then copy out the data vectors from this new state (getXValues())
        # then you replace your state's vector's data fields with these values
        # then you manually or otherwise recalculate the counts and the score --- write a full_score_and_count refresh
        # or do the state-swapping thing, where you switch points
        
        pass
    
    def transition(self,numSteps=1, regen_data=False,time_seatbelt=None,ari_seatbelt=None,true_zs=None):

        time_seatbelt_hit = False
        ari_seatbelt_hit = False
        time_seatbelt_func = (lambda x: False) if time_seatbelt is None else (lambda run_sum: run_sum > time_seatbelt)
        ari_seatbelt_func = (lambda x: False) if ari_seatbelt is None or true_zs is None else (lambda state_zs: hf.calc_ari(state_zs,true_zs)> ari_seatbelt)
        
        for counter in range(numSteps):

            self.transition_beta(time_seatbelt=time_seatbelt) ##may do nothing if infer_beta == "FIXED"
            
            self.transition_z(time_seatbelt=time_seatbelt)

            self.transition_alpha() ##may do nothing if infer_alpha == "FIXED"

            if regen_data:
                self.transition_x()

            if time_seatbelt_func(self.state.timing["run_sum"]):
                time_seatbelt_hit = True
            if ari_seatbelt_func(self.state.getZIndices()):
                ari_seatbelt_hit = True

            if self.state.verbose:
                hf.printTS("Done iteration: ", self.infer_z_count)
                print "Cycle end score: ",self.state.score
                print "alpha: ",self.state.alpha
                print "mean beta: ",self.state.betas.mean()

            if ari_seatbelt_hit or time_seatbelt_hit:
                return {"ari_seatbelt_hit":ari_seatbelt_hit,"time_seatbelt_hit":time_seatbelt_hit}

        return None
            
    def extract_state_summary(self):
        
        state_dict = {
            "alpha":self.state.alpha
            ,"betas":self.state.betas
            ,"score":self.state.score
            ,"numClusters":len(self.state.cluster_list)
            ,"timing":self.state.timing
            ,"state":self.state.get_flat_dictionary()
            }

        if self.calc_ari_func is not None:
            state_dict["ari"] = self.calc_ari_func(self.state.getZIndices())
        else:
            state_dict["ari"] = None

        return state_dict
