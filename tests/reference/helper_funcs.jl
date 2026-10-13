using Printf
using OrderedCollections
using LinearAlgebra
#using LaTeXStrings
#using Plots, Luxor
"""

Naming and ordering conventions:
s (lowercase) : score, generally 0,1,2...
S (uppercase) : has s==0 ('S0') or s>=1 ('S1')
C,N : can help, in need of help
      nb. literature uses d for donor but potentially confuses with actual acts
g,0 : gives help, or declines it

For BigBrothers,
indices will run in the order Need's S, Donor's act, Donor's S.

Ordering within each will go increasing, as in [S0,S1], and [declines,gives].

For strategies,
indices are in the order Receiver_S (being S0 or S1), then Donor_S.
e.g. the strategy named 0,0|g,g gives only to Receivers with positive score (ie. Discriminator)
e.g. the strategy named 0,g|0,g gives (to anyone) if the Donor has positive score.
e.g. the strategy named g,0|0,g gives iff the Donor and Recipient have the SAME score ('gIfSame')
i.e. Strategy goes [[nS0_cS0,nS0_cS1][nS1_cS0,nS1_cS1]]  awesome.
"""

score2str = Dict(0=>"S0", 1=>"S1")
can2index = Dict('0'=>1, 'g'=>2)

function makeAllBB(;USE_CmeetsN=USE_CmeetsN, USE_NmeetsC=USE_NmeetsC, REPUTATION)
			
    """
    Make possible Big Brothers
    Each BB is a Big Brother, represented as a dict with the following keys:
      name  : e.g. `CmeetsN:+0-0+0+- NmeetsC:+0-0+000' 
      CmeetsN : This is a multidim array specifying changes to be made, as chars.
                1st index (outermost brackets) distinguishes cases where
                recipient is S0 (no score) vs S1 (has score). 
              e.g. CmeetsN[1] =  [[nS0_declined_cS0,nS0_declined_cS1],
                                  [nS0_gives_cS0,nS0_gives_cS1],
              and  CmeetsN[2] =  [[nS1_declined_cS0,nS1_declined_cS1],
                                  [nS1_gives_cS0,nS1_gives_cS1],
              where for example 'nS0' refers to a Needy player who has score of 0.  etc etc.
              i.e. 2nd indexes the helping action (declines or gives) 
              and  3rd indexes the CAN player's S.
               ================
              NOTE: Leading 8 notation is:
                    GG - GB - BG - BB 
              BUT NOTE: We represent changes, not new reputation, so this only works for binary. Which is fine, since it's all they have.
              =============
      results: list of ESS strategies, each w stationary score distribution and payoff.
    """
    allBB = OrderedDict{String, Any}()
    # build a list of all the relevant 2x2 change matrices

    if REPUTATION==:binary
        opts_fromS0 = ['0','+']
        opts_fromS1 = ['-','0']
    elseif REPUTATION==:tokens
        opts_fromS0 = ['0','+']
        opts_fromS1 = ['-','0','+']
    elseif REPUTATION==:scores
        opts_fromS0  = ['-','0','+']
        opts_fromS1  = opts_fromS0
    else
    	@printf("Reputation type not recognised\n") 
    end
            
    CmeetsN_options = []
    NmeetsC_options = []

    # CmeetsN
    for nS0_decline_cS0 in opts_fromS0
    for nS0_decline_cS1 in opts_fromS1
    for nS0_give_cS0 in opts_fromS0
    for nS0_give_cS1 in opts_fromS1
    for nS1_decline_cS0 in opts_fromS0
    for nS1_decline_cS1 in opts_fromS1
    for nS1_give_cS0 in opts_fromS0
    for nS1_give_cS1 in opts_fromS1
        push!(CmeetsN_options, Array([[[nS0_decline_cS0,nS0_decline_cS1],[nS0_give_cS0,nS0_give_cS1]],  [[nS1_decline_cS0,nS1_decline_cS1],[nS1_give_cS0,nS1_give_cS1]]]))
    end end end end end end end end
    
    # NmeetsC
    for nS0_decline_cS0 in opts_fromS0
    for nS1_decline_cS0 in opts_fromS1
    for nS0_give_cS0 in opts_fromS0
    for nS1_give_cS0 in opts_fromS1
    for nS0_decline_cS1 in opts_fromS0
    for nS1_decline_cS1 in opts_fromS1
    for nS0_give_cS1 in opts_fromS0
    for nS1_give_cS1 in opts_fromS1
        push!(NmeetsC_options, Array([[[nS0_decline_cS0,nS0_decline_cS1],[nS0_give_cS0,nS0_give_cS1]],  [[nS1_decline_cS0,nS1_decline_cS1],[nS1_give_cS0,nS1_give_cS1]]]))
    end end end end end end end end
    
    # actually scrub those if we are not using them!
    if !USE_CmeetsN
        CmeetsN_options =  [Array([[['0', '0'], ['0', '0']],[['0', '0'], ['0', '0']]])]
    end
    if !USE_NmeetsC
        NmeetsC_options =  [Array([[['0', '0'], ['0', '0']],[['0', '0'], ['0', '0']]])]
    end
    for CmeetsN in CmeetsN_options
    for NmeetsC in NmeetsC_options
            
            BB = Dict{String,Any}()
            BB["CmeetsN"] = CmeetsN  # score chg to the Can, when meet a Need player.
            BB["NmeetsC"] = NmeetsC
            name = bigbrother2name(BB)
            BB["name"] = name
            
            allBB[name] = BB  # puts BB into an allBB dictionary
    end
    end 
    return sort(allBB)
end

function bigbrother2name(bb)
    CmeetsN = replace(string(bb["CmeetsN"]), r"[^-^0^+]" =>"")
    NmeetsC = replace(string(bb["NmeetsC"]), r"[^-^0^+]" =>"")
    return("CmeetsN:"*CmeetsN * " NmeetsC:"*NmeetsC)
end


function name2bigbrother(name)
    bb = Dict{String,Any}()
    elts = split(name)
    for elt in elts
        eltName, str = split(elt,":")
        if eltName in ["CmeetsN","NmeetsC"]
            bb[eltName] = Array([[[str[1],str[2]],[str[3],str[4]]], [[str[5],str[6]],[str[7],str[8]]]])
        end 
    end
    bb["name"] = name
    return(bb)
end

    

##############################################################################

function makeAllstrategies(keepAllStrats=true)
    # make all 16 strategies (ie. strategies for individuals)
    # Returns them as a SortedDict, with the name as key.
    #allstrategies = OrderedDict{String,Any}()
    allstrategies = OrderedDict{String,Any}()
    Help_options = ['0', 'g']
    strategies = Dict{String,Any}()
    todrop=["g,g|0,0","g,0|0,0","g,g|0,g","0,0|g,0","g,0|g,0","g,g|g,0","g,g|0,g"]
    # go through all the options (should be 16)
    for nS0cS0 in Help_options
    for nS0cS1 in Help_options
    for nS1cS0 in Help_options
    for nS1cS1 in Help_options
        strategies = Dict([("strategy",Dict{String,Any}()), ("name","Unnamed")])
        strategies["strategy"]["S0"] = Dict([("S0",nS0cS0),("S1",nS0cS1)])
        strategies["strategy"]["S1"] = Dict([("S0",nS1cS0),("S1",nS1cS1)])
        strategies["name"] = strategy2name(strategies) # knows its own name
        # This is a shortcut to say we would remove these as symmetries later. So let's save computation up front
        if keepAllStrats 
                allstrategies[strategies["name"]] = strategies # name is its key in allstrategies
        else
                if !(strategies["name"] in todrop)
                    allstrategies[strategies["name"]] = strategies 
                end
        end
    end
    end
    end
    end
    allstrategies = sort(allstrategies)
    #for (name,spec) in collect(allstrategies) println(name) end
    return(allstrategies)
end


#------------------------------------------------------------------------------------------------------------
"""
Take a strategy (essentially, a dictionary of offers) and generate a single string that represents it.
"""
function strategy2name(strategies)
    S0name = strategies["strategy"]["S0"]["S0"] * ',' * strategies["strategy"]["S0"]["S1"]
    S1name = strategies["strategy"]["S1"]["S0"] * ',' * strategies["strategy"]["S1"]["S1"]
    return( S0name * "|" * S1name )
end


"""
Take a string representation of a strategy, and generate the full strategy (ie. a dictionary of offers).
"""
function name2strategy(name::String)
    @assert(length(name) == 7)
    strategies = Dict{String,Any}()
    strategies = Dict([("strategy",Dict{String,Any}()), ("name","Unnamed")])
    strategies["strategy"]["S0"] = Dict([("S0",name[1]),("S1",name[3])])
    strategies["strategy"]["S1"] = Dict([("S0",name[5]),("S1",name[7])])
    strategies["name"] = name
    return strategies
end


#------------------------------------------------------------------------------------------------------------

function build_interaction_cases(X, Y, BigB) 
# X,Y being 2 strategies 
# BB being a big brother rule set.
    """
    Precomputation that is useful for the score and payoff calculations to come.
    Consider the interaction between an individual playing strategy X with an 
    individual playing strategy Y and return the cases for each alpha-delta of first player (the one playing X)
    We step through all the 16 options for the 'case'=(Sx,Sy,Cx,Cy), where 
      Sx is 1 if the X player is `solvent', ie. has score above zero.
      Cx is 1 if player is in `CAN` (== in surplus), and 0 if it is in Need;
      And similarly for the Y player.
    The strategies dictate what happens in each such case. 

    If Sx=0 AND (for the X player) ds= +1: include the case (Sx,Sy,Cx,Cy) in a set `alphaCases`
    If Sx=0 AND (for the X player) ds= -1: include the case (Sx,Sy,Cx,Cy) in a set `deltaCases`
    If Sx=1 AND ds= +1: include in `betaCases`
    If Sx=1 AND ds= -1: include in `gammaCases`
    If dh=1 (ie. X player helps): include in 'costCases'
    If dh=-1 (player is helped): include in 'benefitCases'
    
    Return these 6 sets as a single Dict from the name (e.g. 'alphaCases') to the Set.
    """
    xcases = Dict{String,Set{Any}}("alpha"=>Set(), "beta"=>Set(), "gamma"=>Set(), "delta"=>Set(), "cost"=>Set(), "benefit"=>Set())
    

    # This is CmeetsN -- X score updates when they are donor
    Cx,Cy = 1,0  # X is donor, Y is recipient: first case where help may happen.
    for Sx in 0:1 # Thresholded reputation of X (as bad/good)
        for Sy in 0:1 # Thresholded reputation for Y
            strSx, strSy = score2str[Sx], score2str[Sy]
            # is there any help actually given?
            help = X["strategy"][strSy][strSx]
            if (help == 'g')
                push!( xcases["cost"],[Sx, Sy, Cx, Cy]) 
            end
            
            # notice Big Brother also considers the helping action, here.
            # Reminder: Sx=0 will be stored in array elt indexed 1, and
            #           Sx=1 will be stored in array elt indexed 2. 
            # Unfortunate aspect of Julia!
            ds   = BigB["CmeetsN"][Sy+1][can2index[help]][Sx+1]
            if (ds == '+')
                if (Sx == 0) 
                    push!( xcases["alpha"],  [Sx, Sy, Cx, Cy]) 
                elseif (Sx == 1) 
                    push!( xcases["beta"],  [Sx, Sy, Cx, Cy]) 
                end
            elseif (ds == '-')   
                if (Sx == 1) # possibly a bit redundant
                    push!( xcases["gamma"],  [Sx, Sy, Cx, Cy])  
                elseif (Sx == 0) 
                    push!( xcases["delta"],  [Sx, Sy, Cx, Cy]) 
                end
            end
        end        
    end

    # This is NmeetsC -- it is how the X player's score gets updated when they are recipient
    Cx,Cy = 0,1  # X is recipient, Y is donor
    for Sx in 0:1
        for Sy in 0:1
            strSx, strSy = score2str[Sx], score2str[Sy]
            # is there any help actually given?
            help = Y["strategy"][strSx][strSy]
            if (help == 'g')
                push!( xcases["benefit"],[Sx, Sy, Cx, Cy]) 
            end
        
            # notice Big Brother also considers the helping, here.
            # Reminder: Sx=0 will be stored in array elt indexed 1, and
            #           Sx=1 will be stored in array elt indexed 2. 
            # Unfortunate aspect of Julia!
            ds   = BigB["NmeetsC"][Sx+1][can2index[help]][Sy+1]
            if (ds == '+')
                if (Sx == 0) 
                    push!( xcases["alpha"],  [Sx, Sy, Cx, Cy]) 
                elseif (Sx == 1) 
                    push!( xcases["beta"],  [Sx, Sy, Cx, Cy]) 
                end
            elseif (ds == '-')   
                if (Sx == 1) # possibly a bit redundant
                    push!( xcases["gamma"],  [Sx, Sy, Cx, Cy])  
                elseif (Sx == 0) 
                    push!( xcases["delta"],  [Sx, Sy, Cx, Cy]) 
                end
            end
        end        
    end

    #for (k,v) in xcases
    #    println(k,":")
    #    for case in v println("\t",case) end
    #end

    return xcases
end

"""
    Find stable score distribution of an infinite population 
    consisting of a single strategy.
    NOTE: I'm assuming that w_init has catered for negatives as well (so has at least 3 entries)
"""
function calc_stable_w_1strategies(A, BigB; 
            w_init=[0.5,0.0,0.5], tolerance=1e-5, maxn=100, 
            VERBOSE=false,FORCE_UNTIL_MAXN=false,
            REPUTATION=:tokens,
	    NOISE_VAL=0,BENEFIT=2,COST=1)

    cases = build_interaction_cases(A, A, BigB)
    w = copy(w_init)
    len_w_init = length(w_init)
    
    trans = Dict("alpha"=>0.0, "beta"=>0.0, "gamma"=>0.0, "delta"=>0.0) # store the 4 transition probs as a Dict    
    chg = 1.0
    global counter = 0
    prev_w = w
    while counter < maxn  && (chg > tolerance || FORCE_UNTIL_MAXN)  
        counter += 1
        # wBad is everybody with score 0 or -ve, `bad' in binary
	    # Note: this means the order is w0, w-1, w1, w-2, w2, etc.
        wBad = w[1]
        for i in 2:2:length(w)
            wBad+=w[i]
        end
        #wNeg = wBad-w[1]

    	push!(w,0.0,0.0) # add two more empty bins since the top-possible scores (negative and positive) go up each iteration
        #push!(w,0.0) # adds one more (empty) bin, because the top-possible score goes up each iteration.
        #push!(w,0.0) # adds anothermore (empty) bin for the negatives
        # find alpha, etc... which are CONDITIONED on Sx, so we don't include that prob, just the Sy one.
        # alpha: move from -n up to -n+1, including 0: stop at 1
        # beta: move from n to n+1, starting from 1
        # gamma: move n to n-1, including and stopping at 0
        # delta: move from -n to -n-1, starting from 0

        for key in keys(trans)
            total = 0.0
            for case in cases[key]  # Note: case[2] is Sy
                if case[2]==0 total += wBad else total += (1-wBad) end
            end
            trans[key] = total/4.0 # the 1/4 is the p(Cx) * p(Cy) of this case.
            trans[key] = (1.0-NOISE_VAL)*trans[key] + NOISE_VAL/2.0
        end
    
        ########!!!!!!! Binary !!!!!        
        if REPUTATION==:binary
            trans["beta"] = 0.0
            trans["delta"] = 0.0
        elseif REPUTATION==:tokens
   	    # Scores 0 or above
            trans["delta"] = 0.0
        elseif REPUTATION!=:scores
            print("UNKNOWN REPUTATION!!")
            return
        end

        dw = zeros(length(w))

    	dw[1] += -w[1]*trans["alpha"] - w[1]*trans["delta"] + w[3]*trans["gamma"] + w[2]*trans["alpha"] # change to w0
        # This is w(-1)
        dw[2] += -w[2]*trans["alpha"] - w[2]*trans["delta"] + w[1]*trans["delta"] + w[4]*trans["alpha"] # change to w-1
        if REPUTATION != :scores
            @assert(dw[2]==0)
        end
        # This is w(1)
	    dw[3] += -w[3]*trans["gamma"] - w[3]*trans["beta"] + w[1]*trans["alpha"] + w[5]*trans["gamma"] # change to w1
        
        # Note: update what was there before the two new ends were added
        # Evens are negatives
        for i in 4:2:length(w)-2
            dw[i] += -w[i]*trans["alpha"] - w[i]*trans["delta"] + w[i-2]*trans["delta"] + w[i+2]*trans["alpha"]
        end
        # Odds are positives
        for i in 5:2:length(w)-2
		    dw[i] += -w[i]*trans["gamma"] - w[i]*trans["beta"] + w[i-2]*trans["beta"] + w[i+2]*trans["gamma"]
        end

        # n should be odd, need to deal with last (new) positive and negative case
        dw[end] += w[end-2]*trans["beta"]
        dw[end-1] += w[end-3]*trans["delta"]

        #@printf("%.3f\n",sum(dw))
        if abs(dw[1])>1 println(dw) end
	    #if abs(sum(dw))<0.00001 println(dw) end
        #@assert(abs(sum(dw)) <= 0.00001)
   
        prev_w = w  # stashing so I can use it in the fitness calculation that follows this equilibration loop.
        w = w .+ dw
        w = w ./ sum(w)
        chg = maximum(abs.(dw)) # this is perhaps over the top, but works.
        # Special case: all the bottom bins are empty: 
        # chain is transient, and no need to continue.
        
        if sum(w[1:len_w_init]) == 0.0   
            chg = 0.0
        end
    end
    
    # This is just for a final fitness calculation.
    # To keep it sensible (e.g. if we call it after just 1 iteration say), we should use the original 
    # density over scores that was USED to generate the costs and benefits, not the NEW scores that result from that.
    # Hence the "prev_w".
    w0 = prev_w[1]
    for i in 2:2:length(prev_w)
        w0+=prev_w[i]
    end
    prob = Dict("cost"=>0.0, "benefit"=>0.0)
    for key in keys(prob)
        # no conditioning here, so we get the joint prob: a product of appropriate Sx and Sy probs.
        total = 0.0
        for case in cases[key]  # Note: case[2] is Sy
            pr = 1.0
            if case[1]==0 pr *= w0 else pr *= (1-w0) end
            if case[2]==0 pr *= w0 else pr *= (1-w0) end  # ie. quadratic terms are possible
            # @printf("  increment total prob[%s] by %.3f, in case %s\n",key, pr, case)
            total += pr
        end
        prob[key] = total/4.0 
    end    
    fit = BENEFIT*prob["benefit"] - COST*prob["cost"]

    if VERBOSE
        @printf("\tSTRATEGY: %s BigB: %s \n\tw stable: ", A["name"], BigB["name"]) #[9:16] ) #BigB["CmeetsN"])
        for i in 1:min(5,length(w))  @printf("%.3f  ",w[i])   end
        @printf("...(n=%d)\n", length(w))
        scores = zeros(length(w))
        for i in 2:2:length(w)
            scores[i] = -i/2
            scores[i+1] = i/2
        end
        totalscore = sum(w .* scores)
        @printf("\tmean score:  \t %.4f\n",totalscore)
        @printf("\tmean fitness:\t %.4f\n",fit*4)
    else
        totalscore = 0
    end

        return w, fit, trans, totalscore
    
end

"""
    Find stable score distribution of an infinite population 
    consisting of a mixture of two strategies.
    A and B are two strategies that form the whole of an infinite population.
    rhoA is the relative density of A in that population.
    wA_init is an initial density of scores for A, e.g. [.6, .3, .1] would 
    mean a population with 60% having score=0, 30% have 1, and 10% have 2.
    We run a Markov Chain with two possible end conditions:
    exit if the max change to score densities is less than "tolerance", and
    exit if the number of bins exceeds "maxn".
"""
function calc_stable_w_2strategies(A, B, BigB; 
            rhoA=0.5, wA_init=[0.5,0.0,0.5], wB_init=[0.5,0.0,0.5], 
            tolerance=1e-5, maxn=1000, 
            VERBOSE=false,FORCE_UNTIL_MAXN=false,
	        REPUTATION=:tokens,
	        NOISE_VAL=0,BENEFIT=2,COST=1)
	    
    """
    similar to 1strategies, but with 2 strategies, allowing for their densities.
    Handling the interactions right is extremely finicky.
    # Note: by setting maxn=0, one can evaluate fitness of A immediately.
    """
    chainA, chainB = Dict(), Dict()
    chainA["name"] = A["name"]
    chainB["name"] = B["name"]
    chainA["rho"], chainB["rho"] = rhoA,  1-rhoA
    chainA["self"], chainA["other"] = A, B
    chainB["self"], chainB["other"] = B, A
    chainA["w"] = copy(wA_init)
    chainB["w"] = copy(wB_init)
    for ch in [chainA, chainB]
        ch["cases_withSelf"]  = build_interaction_cases(ch["self"], ch["self"], BigB)
        ch["cases_withOther"] = build_interaction_cases(ch["self"], ch["other"], BigB)
        ch["trans"] = Dict("alpha"=>0.0, "beta"=>0.0, "gamma"=>0.0, "delta"=>0.0)    
        ch["n"] = length(ch["w"])
    end
        
    
    global max_chg = 1.0
    global counter = 0
    # until things stabilise, or someone's scores get crazy big...
    while counter < maxn  && (max_chg > tolerance || FORCE_UNTIL_MAXN)  
        # this WAS: while max(chainA["n"], chainB["n"]) < maxn && etc.
        # but that makes it return too soon should it be started with a long chain already.
        # Really, we'd like it to go around lots of times HERE, regardless of the history of w from before.
        counter += 1
        
        for ch in [chainA, chainB]
            ch["n"]  = length(ch["w"])
            push!(ch["w"],0.0,0.0)
            #push!(ch["w"],0.0)
        end
        
        # find the new transition rates, for both chains
        for ch in [chainA, chainB]
            if ch==chainA chOther=chainB else chOther=chainA end 
            w0, w0Other = ch["w"][1], chOther["w"][1] 
            for i in 2:2:length(ch["w"])
                w0+=ch["w"][i]
            end
            for i in 2:2:length(chOther["w"])
                w0Other+=chOther["w"][i]
            end
            for key in keys(ch["trans"])
                grandTotal = 0.0
                total = 0.0
                for case in ch["cases_withSelf"][key]
                    if case[2]==0 total += w0 else total += (1-w0) end    # Note: case[2] is Sy
                end
                grandTotal += ch["rho"] * total
                total = 0.0
                for case in ch["cases_withOther"][key] 
                    if case[2]==0 total += w0Other else total += (1-w0Other) end    # Note: case[2] is Sy
                end
                grandTotal += chOther["rho"] * total
                #ch["trans"][key] = min(1,(grandTotal+NOISE_VAL)/4.0) # the 1/4 is the p(Cx) * p(Cy) of this case.
                #ch["trans"][key] = ((1.0-NOISE_VAL)*grandTotal + NOISE_VAL/2.0)/4
                ch["trans"][key] = grandTotal/4.0 # the 1/4 is the p(Cx) * p(Cy) of this case.
                ch["trans"][key] = (1.0-NOISE_VAL)*ch["trans"][key] + NOISE_VAL/2.0
                #ch["trans"][key] = min(1,ch["trans"][key]+NOISE_VAL) # Noise
                #if ch["trans"][key] > 0.99 println(ch["trans"][key]) end
                #if ch["trans"][key] < 0 println(ch["trans"][key]) end
                #ch["trans"][key] = min(1,ch["trans"][key]+NOISE_VAL) # Noise
            end
            if ch["trans"]["alpha"]+ch["trans"]["delta"]>1 
                ch["trans"]["alpha"]/(ch["trans"]["alpha"]+ch["trans"]["delta"]+0.01)
                ch["trans"]["delta"]/(ch["trans"]["alpha"]+ch["trans"]["delta"]+0.01) 
            end
            if ch["trans"]["beta"]+ch["trans"]["gamma"]>1 
                ch["trans"]["beta"]/(ch["trans"]["beta"]+ch["trans"]["gamma"]+0.01)
                ch["trans"]["gamma"]/(ch["trans"]["beta"]+ch["trans"]["gamma"]+0.01)
            end
            
            if REPUTATION==:binary
                ########!!!!!!! Binary !!!!!
                ch["trans"]["beta"] = 0.0
                ch["trans"]["delta"] = 0.0
            elseif REPUTATION==:tokens
		        # Scores 0 or above
                ch["trans"]["delta"] = 0.0
            end

        end
        
	# Replaces above for now
        # And now do one step with those transition rates, throughout each chain (these are entirely separable).
        global max_chg=0.0
        for ch in [chainA, chainB]
            w = ch["w"]
            t = ch["trans"] # the four transitions: alpha, gamma, beta, delta
            ch["dw"] = zeros(length(w))
            dw = ch["dw"]
	    dw[1] += -w[1]*t["alpha"] - w[1]*t["delta"] + w[3]*t["gamma"] + w[2]*t["alpha"] # change to w0
            # Actually w(-1)
            dw[2] += -w[2]*t["alpha"] - w[2]*t["delta"] + w[1]*t["delta"] + w[4]*t["alpha"] # change to w-1
            # Actually w(1)
	    dw[3] += -w[3]*t["gamma"] - w[3]*t["beta"] + w[1]*t["alpha"] + w[5]*t["gamma"] # change to w1
            # Note: n is the length of w before the two new ends are added
            # Evens are negatives
            for i in 4:2:ch["n"]
                dw[i] += -w[i]*t["alpha"] -w[i]*t["delta"]   + w[i-2]*t["delta"] + w[i+2]*t["alpha"]
            end
            for i in 5:2:ch["n"]
		    dw[i] += -w[i]*t["gamma"] -w[i]*t["beta"]   + w[i-2]*t["beta"] + w[i+2]*t["gamma"]
            end
            # n should be odd, need to deal with last (new) positive and negative case
            dw[end-1] += w[end-3]*t["delta"]
            dw[end] += w[end-2]*t["beta"]
            #if abs(dw[1])>1 println(dw) end
            #if (sum(dw)>0.001) println(dw) end
            #@assert(abs(sum(dw)) <= 0.001)

            w = w .+ dw
            w = w ./ sum(w)
            max_chg = max(max_chg,  maximum( abs.(dw) ))
            ch["w"] = w # NECESSARY, as we haven't changed the actual chain yet.
        end
        if sum(chainA["w"][1:length(wA_init)]) == 0.0  && sum(chainB["w"][1:length(wB_init)]) == 0.0
            max_chg = 0.0 # Special case: all the 'bottom' (close to 0) bins are empty, for both strategies. Both chains transient, no need to continue.
        end
    end


    ################# calculate the (final) fitnesses.
    for ch in [chainA, chainB]
        if ch==chainA chOther=chainB else chOther=chainA end
        w0, w0Other = ch["w"][1], chOther["w"][1] 
         for i in 2:2:length(ch["w"])
             w0+=ch["w"][i]
         end
         for i in 2:2:length(chOther["w"])
             w0Other+=chOther["w"][i]
         end
        prob = Dict("cost"=>0.0, "benefit"=>0.0)
        for key in ["cost", "benefit"]
            # no conditioning here, so calc joint prob: a product of the prob(Sx) and prob(Sy).
            grandTotal = 0.0
            # first, encounters with Self, meaning the same strategy
            total = 0.0
            for case in ch["cases_withSelf"][key]
                pr = 1.0
                if case[1]==0 pr *= w0 else pr *= (1-w0) end  # Note: case[1] is Sx, ie. Self
                if case[2]==0 pr *= w0 else pr *= (1-w0) end  # Note: case[2] is Sy, ie. also Self
                total += pr
            end
            grandTotal += total * ch["rho"]
            # now, encounters with Other
            total = 0.0
            for case in ch["cases_withOther"][key]
                pr = 1.0 
                if case[1]==0 pr *= w0 else pr *= (1-w0) end           # Note: case[1] is Sx, ie. Self
                if case[2]==0 pr *= w0Other else pr *= (1-w0Other) end # Note: case[2] is Sy, ie. Other
                total += pr
            end
            grandTotal += total * chOther["rho"]
            prob[key] = grandTotal/4.0 # the 1/4 is the p(Cx) * p(Cy) of this case.
        end
        #@printf("A: prob[benefit] %.3f \t prob[cost] %.3f \n", prob["benefit"], prob["cost"])
        #@printf("probs (2): %f \t %f \n",prob["benefit"], prob["cost"])
        ch["fit"] = BENEFIT*prob["benefit"] - COST*prob["cost"]
    end
    
    
    if VERBOSE
        for ch in [chainA, chainB]
            @printf("STRATEGY: %s (rho=%.3f) BigB: %s \nw stable: ", ch["name"], ch["rho"], BigB["name"][9:16])
            for i in 1:min(5,length(ch["w"]))  @printf("%.3f  ",ch["w"][i])   end
            @printf("... (n=%d)\n", ch["n"])
            scores = zeros(length(ch["w"]))
            for i in 2:2:(length(ch["w"]))
                scores[i] = -i/2
                scores[i+1] = i/2
            end
            @printf("mean score:  \t %.4f\n",sum(ch["w"] .* scores)) #offset: julia ix from 1.
            @printf("mean fitness:\t %.4f\n\n",ch["fit"]*4)
        end   
        println("=========") 
    end
    return chainA, chainB # these 2 dicts already contain the three things (w,n,fit) we want, so can return them directly.
end

function find_ESS(ESSstrategiesOptionsList, invaderOptionsList, BigB; 
                w_init=w_init, rhoA=0.99, threshold_fitness=1e-5, maxn=100,
                VERBOSE=false,REPUTATION=:tokens,NOISE_VAL=0.01,
		BENEFIT=BENEFIT,COST=COST,careAboutESSSet=false)
    """
    Find all the ESS strategies out of the supplied ESSstrategiesOptionsList, 
    under some particular Big Brother score-manager.
    ESS means resistant to invasion by any of the strategies in invaderOptionsList.
    For full generality, make both these the full sets.
    """
    global c=0
    all_the_ESS = []
    for stratA in ESSstrategiesOptionsList 
        nameA = strategy2name(stratA)
		
        # check it's somewhat cooperative first - can be omitted.
        # Only done to improve speed by skipping the main ESS calculation
        w, fit, trans, totalscore = calc_stable_w_1strategies(
                            stratA, BigB;
                            w_init=w_init, maxn=maxn,
                            REPUTATION=REPUTATION,
			    NOISE_VAL=NOISE_VAL,
			    BENEFIT=BENEFIT,COST=COST)
        if VERBOSE println("Fitness inside findESS is \n",fit*4,nameA,BigB["name"]) end
       
        if (fit*4 < threshold_fitness) continue  end    
        # NOTE: the *4 is because true fitnesses max to 0.25 (prob of C meets N), whereas 1 
        # is a more natural maximum. Hence elsewhere where fitness is written out, it's also
        # multiplied by 4.
        global isESS = true
        global isInESSset = true
        global c += 1
        #@printf("\r \t %8d \t %s   ",c,nameA)   # just for fun: watching the loop

        equalToA = [] # those strategies that are *potentially* in an ESS set with A.
        # If A is eventually found not to be an ESS, this is ditched...
        # Maynard-Smith and Price criteria for ESS is that, for *every* B != A,
        # EITHER 
        # E(A,A) >  E(B,A)  ie. a rare B would be actively die out in a popn of A. 
        # OR
        # E(A,A) == E(B,A)  AND   E(A,B) > E(B,B), i.e. B matches A in a popn of A, but 
        #                                          A would invade pure B from rare.
        #
        #bigList = collect(keys(invaderOptions)) 
        for stratB in invaderOptionsList
            nameB = strategy2name(stratB)

            if nameB != nameA
                 resultA, resultB = 
                 calc_stable_w_2strategies(
                        stratA, stratB, BigB; 
                        wA_init=w_init, wB_init=w_init, rhoA=rhoA, 
                        maxn=maxn, VERBOSE=false,
                        REPUTATION=REPUTATION,
			NOISE_VAL=NOISE_VAL,
			BENEFIT=BENEFIT,COST=COST)
                #if VERBOSE @printf("%s %s %s %.3f %.3f\n",BigB["name"][9:16], nameA, nameB, resultA["fit"],resultB["fit"]) end
                if (resultA["fit"] > resultB["fit"])
                    continue  # A is safe from rare B, so move on to next B contender
                elseif (resultA["fit"] == resultB["fit"]) # they're equal when A common, so we need more evidence...
                    newresultA, newresultB = 
                    calc_stable_w_2strategies(
                            stratA, stratB, BigB; 
                            wA_init=w_init, wB_init=w_init, 
                            rhoA=1-rhoA, ## Note!! this is a check that reverses things.
                            maxn=maxn, VERBOSE=false,
                            REPUTATION=REPUTATION,
			    NOISE_VAL=NOISE_VAL,
			    BENEFIT=BENEFIT,COST=COST)

                    if (newresultA["fit"] > newresultB["fit"])
                        
                        continue # Although common A tolerates some B, common B would 
                                 # lose to rare A, so A's still safe: move on to next B contender.
                    elseif (newresultA["fit"] == newresultB["fit"]) # ie. they're equal, regardless of who is common.
                        if careAboutESSSet
                            println("Nb. TWO EQUAL PAYOFFS CASE")
                            global isESS = false
                            if (newresultA["fit"] > 0) # we can add this to those equal to A.
                                         # nb. A is not strictly an ESS, but might be part of ESS set.
                                push!(equalToA, nameB)
                                #@printf("\t mutual with some (eg) %s \t(fit=%.3f) %s,\n",nameB,newresultA["fit"], BigB["name"])
                                continue # since we do want to collect the set for a look...
                            else 
                                break # just ditch this A and move on.
                            end
                        else
                            continue
                        end
                    else  # ie. A sucks! It tolerates some B, yet loses to B if B get going.
                        global isESS = false
                        global isInESSset = false
                        break
                    end
                else # this is the case that rare B actively beats up common A, so A isn't ESS.
                    if VERBOSE @printf("\t actively invaded by (eg) %s\n",nameB) end
                    global isESS = false
                    global isInESSset = false
                    break
                end
            end        
        end
        
        # we've checked everything -- report back if it really is an ESS
        if isESS 
            if VERBOSE @printf("\n\n%s is ESS.",nameA) end
                w, fit, trans, totalscore = calc_stable_w_1strategies(stratA, BigB;
                                w_init=w_init, REPUTATION=REPUTATION,
				NOISE_VAL=NOISE_VAL, BENEFIT=BENEFIT, COST=COST)
            if fit < 1e-5
                if VERBOSE @printf("\t But a boring one: fitness is %f\n",fit) end
            else
                if VERBOSE 
                    @printf("\nDETAILS:\n") 
                    calc_stable_w_1strategies(name2strategy(nameA), BigB; 
                                w_init=w_init, VERBOSE=true,
                                REPUTATION=REPUTATION,
			                	NOISE_VAL=NOISE_VAL, BENEFIT=BENEFIT, COST=COST)
                
                    global countThisInvades=0
                    for B in invaderOptionsList
                        local resultA, resultB = 
                        calc_stable_w_2strategies(
                                stratA, B, BigB; 
                                wA_init=w_init, wB_init=w_init,
                                rhoA=1-rhoA, maxn=maxn,
                                REPUTATION=REPUTATION,
		                		NOISE_VAL=NOISE_VAL, BENEFIT=BENEFIT, COST=COST)
                        if resultA["fit"] > resultB["fit"]
                            global countThisInvades += 1
                        end
                    end
                    @printf("\t%s can ACTIVELY invade %d other strats from rare\n\n",nameA,countThisInvades) 
                end
                push!(all_the_ESS, nameA)
            end
        else
            if isInESSset #length(equalToA) > 0
                w, fit, trans, totalscore = calc_stable_w_1strategies(
                        stratA, BigB;
                        w_init=w_init, 
                        REPUTATION=REPUTATION,
			NOISE_VAL=NOISE_VAL,
			BENEFIT=BENEFIT, COST=COST)
                if (fit > 0.0) && VERBOSE
                    @printf("\n\t\tNOT strict ESS, but has fitness %f, is == to these:\n", fit)
                    for z in unique(equalToA)  @printf("\t\t%s\n",z) end
                end
            end
        end
    end
    if VERBOSE 
        @printf("\n\n")
    end
    return all_the_ESS
    
end


# ================================================================
function big_test(;REPUTATION=:tokens,
                        USE_CmeetsN=USE_CmeetsN,  
                        USE_NmeetsC=USE_NmeetsC, 
                        WSTART=WSTART, #[0.5,0.5],
                        maxn=100,rhoA=0.99,noiseval=0.01,
                        threshold_fitness=0.05,strategiesOptionsDict=nothing,
			BENEFIT=BENEFIT,COST=COST)
    
    @assert(REPUTATION==:binary || REPUTATION==:tokens || REPUTATION==:scores)
    keepAllStrategies = (REPUTATION!=:scores)

    invaderOptionsList = collect(values(makeAllstrategies()))
    if strategiesOptionsDict == nothing
        strategiesOptionsDict = makeAllstrategies(keepAllStrategies)
    end
    strategiesOptionsList = collect(values(strategiesOptionsDict))
    
    BigBBs = makeAllBB(;USE_CmeetsN=USE_CmeetsN,  
                        USE_NmeetsC=USE_NmeetsC, 
                        REPUTATION=REPUTATION)
    @printf("# Reputation type: %s, noise value: %.3f, max iterations: %d\n", REPUTATION, noiseval, maxn)
    @printf("# BENEFIT %s, COST %s ", string(BENEFIT), string(COST))
    @printf("# WSTART: %s, \n",string(WSTART))
    #print(collect(values(BigBBs)))
    startTime = time() 
    for spec1 in strategiesOptionsList

        @printf("#\n# %s being considered for ESS.\n",spec1["name"])
	@printf("# BBname (CmeetsN,NmeetsC,.. Hilbe) ESS(don) fit w0 w-1 w1 α β γ δ\n")

        for BigB in collect(values(BigBBs))
            all_the_ESS =  find_ESS([spec1], invaderOptionsList, BigB; 
                                    w_init=WSTART, rhoA=rhoA, 
                                    threshold_fitness=threshold_fitness, 
                                    maxn=maxn, 
                                    VERBOSE=false, 
                                    REPUTATION=REPUTATION,
                                    NOISE_VAL=noiseval,
			            BENEFIT=BENEFIT,COST=COST)
                                    
            if length(all_the_ESS) > 0
                subs = split(BigB["name"],[' ',':'])
                for ESSname in all_the_ESS
                    w, fit, trans, totalscore = calc_stable_w_1strategies(strategiesOptionsDict[ESSname], BigB;
					    w_init=WSTART,maxn=maxn,REPUTATION=REPUTATION, NOISE_VAL=noiseval,
			                    BENEFIT=BENEFIT,COST=COST)
                    #@printf("\t%s  fit:%.3f  w0:%.3f  w1:%.3f",name,fit,w[1],w[2])
                    #@printf("  α:%.3f  β:%.3f  γ:%.3f\n",trans["alpha"],trans["beta"],trans["gamma"])
                    #if INCLUDE_HILBE_NOTATION
                        #print(subs[2], "\t", BigB["name"])
                        #tmp  = convertToHilbeNotation(BigB["name"][7:14])  #CmeetsN rule, how donor scores change
                        #tmp2 = convertToHilbeNotation(BigB["name"][22:29]) #NmeetsC rule, how recvr scores change
                        #@printf("%s %s %s %s ",subs[2],subs[4],tmp*" "*tmp2, ESSname)
                    #else
                    @printf("%s %s %s ",subs[2],subs[4], ESSname)                
                    #end
                    @printf("%.3f ",fit*4)
                    @printf("%.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",  w[1],w[2],w[3],trans["alpha"],trans["beta"],trans["gamma"],trans["delta"])
			# Old version prints without the delta
                        #@printf("%.3f %.3f %.3f %.3f %.3f %.3f ", 
                            #fit*4,w[1],w[2],trans["alpha"],trans["beta"],trans["gamma"])
                    
                    # Compute the flux, for every entry in BigB, and print them all out on one line.
                    # Notice: passing in w0 as it's already known and avoids re-running the whole chain.
                    #dum1,dum2,dum3,dum4 = printEqlbmFluxes(ESSname, BigB["name"], w[begin])
                                        
                end
            end
        end
    end
    @printf("#\n# That took %.0f seconds\n#\n",time() - startTime)
end

function small_test(strategyOptionsList,thisBB; REPUTATION=:tokens,
                    WSTART=[0.5,0,0.5], maxn=100, rhoA=0.99, 
		            noiseval=0.01, threshold_fitness=0.01,
		            BENEFIT=2,COST=1,VERBOSE=false)
	
    @assert(REPUTATION==:binary || REPUTATION==:tokens || REPUTATION==:scores)
    #strategyOptionsDict =  makeAllstrategies()
    #strategyOptionsList = collect(values(strategyOptionsDict))
    
    all_the_ESS =  find_ESS(strategyOptionsList, strategyOptionsList, thisBB; 
                            w_init=WSTART, rhoA=rhoA, 
                            threshold_fitness=threshold_fitness, 
                            maxn=maxn,  VERBOSE=VERBOSE, 
                            REPUTATION=REPUTATION,
                            NOISE_VAL=noiseval,
			    BENEFIT=BENEFIT,COST=COST)
    		        
    if length(all_the_ESS) > 0  # if so, print the details...
		best_fitness = threshold_fitness
		best_ESSname = nothing
                best_score = nothing
    
        for ESSname in all_the_ESS
                
                w, fit, trans, totalscore = calc_stable_w_1strategies(
								strategyOptionsDict[ESSname], thisBB;
								w_init=WSTART,	maxn=maxn,
								REPUTATION=REPUTATION, VERBOSE=VERBOSE,
								NOISE_VAL=noiseval, BENEFIT=BENEFIT, COST=COST)
                fit = fit*4 # this is a bit silly TBF.
			
		if fit > best_fitness
			best_fitness = fit
			best_ESSname = ESSname
                        best_score = totalscore
		end
        end
		return best_ESSname, best_fitness, best_score
    end
	return nothing, -1.0, -1.0
end

function small_test_back(strategyOptionsList,thisBB; REPUTATION=:tokens,
                    WSTART=[0.5,0,0.5], maxn=100, rhoA=0.99, 
		            noiseval=0.01, threshold_fitness=0.01,
		            BENEFIT=2,COST=1,VERBOSE=false)
	
    #strategyOptionsDict =  makeAllstrategies()
    #strategyOptionsList = collect(values(strategyOptionsDict))
    
    all_the_ESS =  find_ESS(strategyOptionsList, strategyOptionsList, thisBB; 
                            w_init=WSTART, rhoA=rhoA, 
                            threshold_fitness=0.01, 
                            maxn=maxn,  VERBOSE=VERBOSE, 
                            REPUTATION=REPUTATION,
                            NOISE_VAL=noiseval,
			                BENEFIT=BENEFIT,COST=COST)
    		        
    if length(all_the_ESS) > 0  # if so, print the details...
		best_fitness = 0 # threshold_fitness
		best_ESSname = nothing
        for ESSname in all_the_ESS
                w, fit, trans = calc_stable_w_1strategies(
								strategyOptionsDict[ESSname], thisBB;
								w_init=WSTART,	maxn=maxn,
								REPUTATION=REPUTATION,
								NOISE_VAL=noiseval,
			    				BENEFIT=BENEFIT,COST=COST)
		fit = fit*4 # this is a bit silly TBF.
			
                # Two options: get the highest fitness below the current, or the lowest
                        # 1st:
		if fit < threshold_fitness && fit > best_fitness
                #set best_fitness above to 0, not threshold_fitness
			best_fitness = fit
			best_ESSname = ESSname
                        #print(fit,'\n')
		end

                        # 2nd:
                #set best_fitness above to threshold_fitness, not 0
		#if fit < best_fitness && fit > -1
			#best_fitness = fit
			#best_ESSname = ESSname
		#end
        end
        #print("--- \n")
		return best_ESSname, best_fitness
    end
	return nothing, -1.0
end

###########################################

# check some basic functions work as intended
function test_string2name_functions()
    testBBname = "CmeetsN:+00+-++- NmeetsC:+00+-+00" 
    testBBname = "CmeetsN:000000++ NmeetsC:000000--" # money
    @assert(bigbrother2name(name2bigbrother(testBBname)) == testBBname)

    #=
    bbs = makeAllBB()
    testBBname = collect(keys(bbs))[666666]
    testBB = bbs[testBBname]
    @assert(bigbrother2name(testBB) == testBBname)
    =#
    
    for teststrat in ["0,g|0,g",  "g,0|0,g"]
        @assert(strategy2name(name2strategy(teststrat)) == teststrat)
    end
end

