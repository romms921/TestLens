#!/usr/bin/env python
import glafic
glafic.init(0.3, 0.7, -1.0, 0.7, 'out/SIE_POS', -3.5, -3.5, 3.5, 3.5, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 0', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    0', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(1, 0, 1)
glafic.set_lens(1, 'sie', 0.2300, 2.368933e+02,  -0.832, 1.220, 3.203552e-01, 0, 0.000000e+00, 0.0)
glafic.set_point(1, 0.777, 0.0, 0.0)

glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/TestLens/obs/obs_point_SIE(POS).dat') # Enter path to obs_point file

glafic.optimize()
glafic.findimg()
glafic.writecrit(0.777)
glafic.writelens(0.777)

glafic.quit()
