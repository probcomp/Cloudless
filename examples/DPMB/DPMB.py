#!python
import datetime,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
reload(ds)
import helper_functions as hf
reload(hf)
##
import pdb


class DPMB():
    def __init__(self,inf_seed,state,infer_alpha,infer_beta):
        hf.set_seed(inf_seed)
        self.state = state
        self.infer_alpha = infer_alpha
        self.infer_beta = infer_beta
        ##
        self.transition_count = 0
        self.time_seatbelt_hit = False
        self.ari_seatbelt_hit = False
    
    def transition_alpha_discrete_gibbs(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        if self.check_time_seatbelt(time_seatbelt):
            return # don't transition
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

            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break

            logp_list, lnPdf, grid = hf.calc_beta_conditional(self.state,col_idx)
            beta_idx = hf.renormalize_and_sample(logp_list)
            self.state.removeBetaD(lnPdf,col_idx)
            self.state.setBetaD(lnPdf,col_idx,grid[beta_idx])

        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).seconds()
        self.state.timing["run_sum"] += self.state.timing["betas"]

    def transition_alpha(self,time_seatbelt=None):
        if self.state.verbose:
            print "PRE transition_alpha score: ",self.state.score
        if self.infer_alpha:
            self.transition_alpha_discrete_gibbs(time_seatbelt=None)
        else:
            self.state.timing["alpha"] = 0 ##ensure last value not carried forward

    def transition_beta(self,time_seatbelt=None):
        if self.state.verbose:
            print "PRE transition_beta score: ",self.state.score
        if self.infer_beta:
            self.transition_beta_discrete_gibbs(time_seatbelt)
        else:
            self.state.timing["betas"] = 0 ##ensure last value not carried forward
            
    def transition_z(self,time_seatbelt=None):
        if self.state.verbose:
            print "PRE transition_z score: ",self.state.score
        start_dt = datetime.datetime.now()
        ##
        # for each vector
        for vector in nr.permutation(self.state.get_all_vectors()):

            if self.state.verbose:
                print " - transitioning vector idx " + str(self.state.vector_list.index(vector))

            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break

            # deassign it
            vector.cluster.deassign_vector(vector)
            
            # calculate the conditional
            score_vec = hf.calculate_cluster_conditional(self.state,vector)

            # sample an assignment

            draw = hf.renormalize_and_sample(score_vec,verbose=self.state.verbose)

            cluster = None
            if draw == len(self.state.cluster_list):
                cluster = self.state.generate_cluster_assignment(force_new = True)
            else:
                cluster = self.state.cluster_list[draw]

            # assign it
            cluster.assign_vector(vector)

        # debug print out states:
        if self.state.verbose or True:
            # print " --- " + str(self.state.getZIndices())
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

        for counter in range(numSteps):

            self.transition_beta(time_seatbelt=time_seatbelt) ##may do nothing if infer_beta == "FIXED"
            
            self.transition_z(time_seatbelt=time_seatbelt)

            self.transition_alpha(time_seatbelt=time_seatbelt) ##may do nothing if infer_alpha == "FIXED"

            if regen_data:
                self.transition_x()
                
            self.check_time_seatbelt(time_seatbelt)
            self.check_ari_seatbelt(ari_seatbelt,true_zs)

            if self.state.verbose:
                hf.printTS("Done iteration: ", self.transition_count)
                print "Cycle end score: ",self.state.score
                print "alpha: ",self.state.alpha
                print "mean beta: ",self.state.betas.mean()

            if self.ari_seatbelt_hit or self.time_seatbelt_hit:
                return {"ari_seatbelt_hit":self.ari_seatbelt_hit,"time_seatbelt_hit":self.time_seatbelt_hit}

            self.transition_count += 1

        return None

    def check_time_seatbelt(self,time_seatbelt=None,delta_t=0):
        if time_seatbelt is None:
            return self.time_seatbelt_hit
        self.time_seatbelt_hit = self.state.timing["run_sum"] + delta_t > time_seatbelt
        return self.time_seatbelt_hit

    def check_ari_seatbelt(self,ari_seatbelt=None,true_zs=None):
        if ari_seatbelt is None or true_zs is None:
            return self.ari_seatbelt_hit
        self.ari_seatbelt_hit = hf.calc_ari(self.state.getZIndices(),true_zs) > ari_seatbelt
        return self.ari_seatbelt_hit

    def extract_state_summary(self,true_zs=None,send_zs=True,verbose_state=False,test_xs=None):
        
        state_dict = {
            "alpha":self.state.alpha
            ,"betas":self.state.betas.copy()
            ,"score":self.state.score
            ,"num_clusters":len(self.state.cluster_list)
            ,"timing":self.state.get_timing()
            ,"inf_seed":hf.get_seed()
            }

        if true_zs is not None:
            state_dict["ari"] = hf.calc_ari(true_zs,self.state.getZIndices())
        else:
            state_dict["ari"] = None

        if verbose_state or send_zs:
            state_dict["zs"] = self.state.getZIndices()
        if verbose_state:
            state_dict["xs"] = self.state.getXValues()

        if test_xs is not None:
            state_dict["test_lls"] = self.state.score_test_set(test_xs)
        return state_dict