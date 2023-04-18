# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from run_manager import *

from torch.profiler import profile, record_function, ProfilerActivity

class ArchSearchConfig:

    def __init__(self, arch_init_type, arch_init_ratio, arch_opt_type, arch_lr,
                 arch_opt_param, arch_weight_decay, target_hardware, ref_value):
        """ architecture parameters initialization & optimizer """
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.opt_type = arch_opt_type
        self.lr = arch_lr
        self.opt_param = {} if arch_opt_param is None else arch_opt_param
        self.weight_decay = arch_weight_decay
        self.target_hardware = target_hardware
        self.ref_value = ref_value

    @property
    def config(self):
        config = {
            'type': type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, nBatch):
        raise NotImplementedError

    def build_optimizer(self, params):
        """

        :param params: architecture parameters
        :return: arch_optimizer
        """
        if self.opt_type == 'adam':
            return torch.optim.Adam(
                params, self.lr, weight_decay=self.weight_decay, **self.opt_param
            )
        else:
            raise NotImplementedError


class GradientArchSearchConfig(ArchSearchConfig):

    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, target_hardware=None, ref_value=None,
                 grad_update_arch_param_every=1, grad_update_steps=1, grad_binary_mode='full', grad_data_batch=None,
                 grad_reg_loss_type=None, grad_reg_loss_params=None, **kwargs):
        super(GradientArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_opt_type, arch_lr, arch_opt_param, arch_weight_decay,
            target_hardware, ref_value,
        )

        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode
        self.data_batch = grad_data_batch

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params
        
        print(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.update_steps
        return schedule

    def add_regularization_loss(self, ce_loss, expected_value, block_idx = None):
        if expected_value is None:
            return ce_loss
        
        if block_idx is not None:
            ref_value = self.ref_value[block_idx]
        else:
            ref_value = self.ref_value 

        if self.reg_loss_type == 'mul#log':
            alpha = self.reg_loss_params.get('alpha', 1)
            beta = self.reg_loss_params.get('beta', 0.6)
            
            # noinspection PyUnresolvedReferences
            reg_loss = (torch.log(expected_value) / math.log(ref_value)) ** beta

            if torch.distributed.get_rank() == 0:
                print('expected_value:',expected_value)
                print('ref_value:', ref_value)
                print('ce_loss:',ce_loss)
                print('reg_loss:',reg_loss)

            return alpha * ce_loss * reg_loss
        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_value - ref_value) / ref_value
            return ce_loss + reg_loss
        elif self.reg_loss_type is None:
            return ce_loss
        else:
            raise ValueError('Do not support: %s' % self.reg_loss_type)


class RLArchSearchConfig(ArchSearchConfig):

    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, target_hardware=None, ref_value=None,
                 rl_batch_size=10, rl_update_per_epoch=False, rl_update_steps_per_epoch=300,
                 rl_baseline_decay_weight=0.99, rl_tradeoff_ratio=0.1, **kwargs):
        super(RLArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_opt_type, arch_lr, arch_opt_param, arch_weight_decay,
            target_hardware, ref_value,
        )

        self.batch_size = rl_batch_size
        self.update_per_epoch = rl_update_per_epoch
        self.update_steps_per_epoch = rl_update_steps_per_epoch
        self.baseline_decay_weight = rl_baseline_decay_weight
        self.tradeoff_ratio = rl_tradeoff_ratio

        self._baseline = None
        print(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        if self.update_per_epoch:
            schedule[nBatch - 1] = self.update_steps_per_epoch
        else:
            rl_seg_list = get_split_list(nBatch, self.update_steps_per_epoch)
            for j in range(1, len(rl_seg_list)):
                rl_seg_list[j] += rl_seg_list[j - 1]
            for j in rl_seg_list:
                schedule[j - 1] = 1
        return schedule

    def calculate_reward(self, net_info):
        acc = net_info['acc'] / 100
        if self.target_hardware is None:
            return acc
        else:
            return acc * ((self.ref_value / net_info[self.target_hardware]) ** self.tradeoff_ratio)

    @property
    def baseline(self):
        return self._baseline

    @baseline.setter
    def baseline(self, value):
        self._baseline = value


class ArchSearchRunManager:

    def __init__(self, path, super_net, run_config: RunConfig, arch_search_config: ArchSearchConfig, out_log = True, stage = 0, rank_to_feature_shape = None, rank_to_pgroup = None, teacher_net=None, skip_init = False):
        
        self.arch_search_config = arch_search_config

        # init architecture parameters
        if skip_init == False:
            super_net.init_arch_params(
                self.arch_search_config.arch_init_type, self.arch_search_config.arch_init_ratio,
            )

                
        # init weight parameters & build weight_optimizer
        self.run_manager = RunManager(path, super_net, run_config, out_log, stage, rank_to_feature_shape, rank_to_pgroup, skip_init)

        # build architecture optimizer
        self.arch_optimizer = self.arch_search_config.build_optimizer(super_net.architecture_parameters())

        self.warmup = True
        self.warmup_epoch = 0

        """teacher_net is added"""
        self.teacher_net = teacher_net 
        if self.teacher_net is not None:
            if self.run_manager.run_config.train_method == 'dp' or self.run_manager.run_config.train_method == None:
                assert(self.teacher_net is None)
            elif self.run_manager.run_config.train_method == 'dna':
                self.teacher_net = self.teacher_net.to(self.run_manager.device)
            elif self.run_manager.run_config.train_method == 'mp':
                self.teacher_net= self.teacher_net.to(self.run_manager.device)
            elif self.run_manager.run_config.train_method == 'ts':
                self.teacher_net= self.teacher_net.to(self.run_manager.device)
            else:
                print(torch.distributed.get_rank(),'has no teacher!')
                raise NotImplementedError

    @property
    def net(self):
        if isinstance(self.run_manager.net, DDP):
            return self.run_manager.net.module
        else:
            return self.run_manager.net

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))

        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname, map_location='cpu')
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict)
        if self.run_manager.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))

        # set new manual seed
        #new_manual_seed = int(time.time())
        #torch.manual_seed(new_manual_seed)
        #torch.cuda.manual_seed_all(new_manual_seed)
        #np.random.seed(new_manual_seed)

        if 'epoch' in checkpoint:
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
        if 'weight_optimizer' in checkpoint:
            self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        if 'arch_optimizer' in checkpoint:
            self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        if 'warmup' in checkpoint:
            self.warmup = checkpoint['warmup']
        if self.warmup and 'warmup_epoch' in checkpoint:
            self.warmup_epoch = checkpoint['warmup_epoch']

    """ training related methods """

    def validate(self):
        # get performances of current chosen network on validation set
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # set chosen op active
        self.net.set_chosen_op_active()
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        valid_res = self.run_manager.validate(is_test=False, use_train_mode=True, return_top5=True)
        # flops of chosen network
        flops = self.run_manager.net_flops()
        # measure latencies of chosen op
        if self.arch_search_config.target_hardware in [None, 'flops']:
            latency = 0
        else:
            latency, _ = self.run_manager.net_latency(
                l_type=self.arch_search_config.target_hardware, fast=False
            )
        # unused modules back
        self.net.unused_modules_back()
        return valid_res, flops, latency

    def warm_up(self, warmup_epochs=25):
        lr_max = 0.05
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch

        for epoch in range(self.warmup_epoch, warmup_epochs):
            if self.run_manager.out_log:
                print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                end = time.time()
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.run_manager.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                # compute output
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup
                output = self.run_manager.net(images)  # forward (DataParallel)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    loss = self.run_manager.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                # measure elapsed time
                torch.cuda.synchronize()
                batch_time.update(time.time() - end)
                
                reduced_loss = reduce_tensor(loss.data, self.run_manager.world_size)
                acc1 = reduce_tensor(acc1, self.run_manager.world_size)
                acc5 = reduce_tensor(acc5, self.run_manager.world_size)
                losses.update(reduced_loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                if self.run_manager.out_log and (i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch):
                    
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=warmup_lr)
                    self.run_manager.write_log(batch_log, 'train')
                end = time.time()
            valid_res, flops, latency = self.validate()
            if self.run_manager.out_log:
                val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                          'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\tflops: {5:.1f}M'. \
                    format(epoch + 1, warmup_epochs, *valid_res, flops / 1e6, top1=top1, top5=top5)
                if self.arch_search_config.target_hardware not in [None, 'flops']:
                    val_log += '\t' + self.arch_search_config.target_hardware + ': %.3fms' % latency
                self.run_manager.write_log(val_log, 'valid')

            self.warmup = epoch + 1 < warmup_epochs

            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {
                'state_dict': state_dict,
                'warmup': self.warmup,
            }
            if self.warmup:
                checkpoint['warmup_epoch'] = epoch,
            self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')

    def train(self, fix_net_weights=False):
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch

        arch_param_num = len(list(self.net.architecture_parameters()))
        binary_gates_num = len(list(self.net.binary_gates()))
        weight_param_num = len(list(self.net.weight_parameters()))
        if self.run_manager.out_log:
            print(
                '#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
                (arch_param_num, binary_gates_num, weight_param_num)
            )
        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        
        avg_epoch_time = AverageMeter()
        avg_batch_time = AverageMeter()

        avg_data_time = AverageMeter()
        avg_prepare_time = AverageMeter()
        avg_sf_time = AverageMeter()
        avg_sb_time = AverageMeter()
        avg_arch_time = AverageMeter() 


        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
            if self.run_manager.out_log:
                print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

            #time log
            batch_time = AverageMeter()

            data_time = AverageMeter()
            prepare_time = AverageMeter()
            sf_time= AverageMeter()
            sb_time= AverageMeter()
            arch_time = AverageMeter() 

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()

            # switch to train mode
            self.run_manager.net.train()

            epoch_start_time = time.time()
            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                torch.cuda.synchronize()
                data_time.update(time.time() - end)
                end = time.time()
                batch_start = end

                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                # network entropy
                #net_entropy = self.net.entropy()
                #entropy.update(net_entropy.data.item() / arch_param_num, 1)

                # compute output
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup

                prepare_time.update(time.time() -end)
                end =time.time()

                output = self.run_manager.net(images)  # forward (DataParallel)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    loss = self.run_manager.criterion(output, labels)
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                
                # measure elapsed time
                torch.cuda.synchronize()
                sb_time.update(time.time() - end)
                end = time.time()
                
                # skip architecture parameter updates in the first epoch
                #if epoch > 0:
                if epoch >= int((self.run_manager.run_config.n_epochs - 0)/4):
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        if isinstance(self.arch_search_config, RLArchSearchConfig):
                            reward_list, net_info_list = self.rl_update_step(fast=True)
                            used_time = time.time() - start_time
                            log_str = 'REINFORCE [%d-%d]\tTime %.4f\tMean Reward %.4f\t%s' % (
                                epoch + 1, i, used_time, sum(reward_list) / len(reward_list), net_info_list
                            )
                            self.write_log(log_str, prefix='rl', should_print=False)
                        elif isinstance(self.arch_search_config, GradientArchSearchConfig):
                            arch_loss, exp_value = self.gradient_step()
                            if self.run_manager.out_log:
                                torch.cuda.synchronize()
                                arch_time.update(time.time() - end)
                                log_str = 'Architecture [%d-%d]\t Loss %.4f\t %s %s' % \
                                          (epoch + 1, i, arch_loss,
                                           self.arch_search_config.target_hardware, exp_value)
                                self.write_log(log_str, prefix='gradient', should_print=False)
                        else:
                            raise ValueError('do not support: %s' % type(self.arch_search_config))
                
                batch_time.update(time.time() - batch_start)
               
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                reduced_loss = reduce_tensor(loss.data, self.run_manager.world_size)
                acc1 = reduce_tensor(acc1, self.run_manager.world_size)
                acc5 = reduce_tensor(acc5, self.run_manager.world_size)
                losses.update(reduced_loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                # training log
                if self.run_manager.out_log and (i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch):
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Loss {losses.val:.4f}({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f}({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f}({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f}({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1,top5=top5,lr=lr)
                    self.run_manager.write_log(batch_log, 'train')
                    
                    batch_log = 'Time [{0}][{1}/{2}]\t' \
                                'B {batch_time.val:.4f}({batch_time.avg:.4f})\t' \
                                'D {data_time.val:.4f}({data_time.avg:.4f})\t' \
                                'P {prepare_time.val:.4f}({prepare_time.avg:.4f})\t' \
                                'SF {sf_time.val:.4f}({sf_time.avg:.4f})\t' \
                                'SB {sb_time.val:.4f}({sb_time.avg:.4f})\t' \
                                'A {arch_time.val:.4f}({arch_time.avg:.4f})\t'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time, prepare_time=prepare_time,
                               sf_time=sf_time, sb_time=sb_time, arch_time= arch_time)
                    self.run_manager.write_log(batch_log, 'time', should_print=False)
                """
                if i == 0: 
                    batch_time = AverageMeter()
                    data_time = AverageMeter()
                    prepare_time = AverageMeter()
                    sf_time= AverageMeter()
                    sb_time= AverageMeter()
                    arch_time = AverageMeter() 
                """
                end = time.time()

            if epoch >= 0:
                avg_epoch_time.update(time.time() - epoch_start_time)
                avg_batch_time.update(batch_time.avg)

                avg_data_time.update(data_time.avg)
                avg_prepare_time.update(prepare_time.avg)
                avg_sf_time.update(sf_time.avg)
                avg_sb_time.update(sb_time.avg)
                avg_arch_time.update(arch_time.avg)

                batch_log = '[{0:>3}]\t' \
                            'B {batch_time.val:.4f}({batch_time.avg:.4f})\t' \
                            'D {data_time.val:.4f}({data_time.avg:.4f})\t' \
                            'P {prepare_time.val:.4f}({prepare_time.avg:.4f})\t' \
                            'SF {sf_time.val:.4f}({sf_time.avg:.4f})\t' \
                            'SB {sb_time.val:.4f}({sb_time.avg:.4f})\t' \
                            'A {arch_time.val:.4f}({arch_time.avg:.4f})\t' \
                            'Epoch {epoch_time.val:.4f}( {epoch_time.avg:.4f})\t'. \
                    format( epoch + 1, epoch_time= avg_epoch_time, batch_time=avg_batch_time, data_time=avg_data_time,
                           prepare_time=avg_prepare_time, sf_time=avg_sf_time, sb_time=avg_sb_time,
                           arch_time =avg_arch_time)
                self.run_manager.write_log(batch_log, 'time', should_print=False)


            # print current network architecture
            if self.run_manager.out_log:
                self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
                for stage in self.net.blocks:
                    for idx, block in enumerate(stage):
                        self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
                self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                if self.run_manager.out_log:
                    self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                    val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                              'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                              'Entropy {entropy.val:.5f}\t' \
                              'Latency-{6}: {7:.3f}ms\tFlops: {8:.2f}M'. \
                        format(epoch + 1, self.run_manager.run_config.n_epochs, val_loss, val_top1,
                               self.run_manager.best_acc, val_top5, self.arch_search_config.target_hardware,
                               latency, flops / 1e6, entropy=entropy, top1=top1, top5=top5)
                    self.run_manager.write_log(val_log, 'valid')
            if self.run_manager.out_log:
                # save model
                self.run_manager.save_model({
                    'warmup': False,
                    'epoch': epoch,
                    'weight_optimizer': self.run_manager.optimizer.state_dict(),
                    'arch_optimizer': self.arch_optimizer.state_dict(),
                    'state_dict': self.net.state_dict()
                })
        
                
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, 'run.config'), 'w'), indent=4,
        )
        if self.run_manager.out_log:
            # convert to normal network according to architecture parameters
            normal_net = self.net.cpu().convert_to_normal_net()
            print('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6))
            os.makedirs(os.path.join(self.run_manager.path, 'learned_net'), exist_ok=True)
            json.dump(normal_net.config, open(os.path.join(self.run_manager.path, 'learned_net/net.config'), 'w'), indent=4)
            torch.save(
                {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
                os.path.join(self.run_manager.path, 'learned_net/init')
            )

    def rl_update_step(self, fast=True):
        assert isinstance(self.arch_search_config, RLArchSearchConfig)
        # prepare data
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
        # sample nets and get their validation accuracy, latency, etc
        grad_buffer = []
        reward_buffer = []
        net_info_buffer = []
        for i in range(self.arch_search_config.batch_size):
            self.net.reset_binary_gates()  # random sample binary gates
            self.net.unused_modules_off()  # remove unused module for speedup
            # validate the sampled network
            with torch.no_grad():
                output = self.run_manager.net(images)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            net_info = {'acc': acc1[0].item()}
            # get additional net info for calculating the reward
            if self.arch_search_config.target_hardware is None:
                pass
            elif self.arch_search_config.target_hardware == 'flops':
                net_info['flops'] = self.run_manager.net_flops()
            else:
                net_info[self.arch_search_config.target_hardware], _ = self.run_manager.net_latency(
                    l_type=self.arch_search_config.target_hardware, fast=fast
                )
            net_info_buffer.append(net_info)
            # calculate reward according to net_info
            reward = self.arch_search_config.calculate_reward(net_info)
            # loss term
            obj_term = 0
            for m in self.net.redundant_modules:
                if m.AP_path_alpha.grad is not None:
                    m.AP_path_alpha.grad.data.zero_()
                obj_term = obj_term + m.log_prob
            loss_term = -obj_term
            # backward
            loss_term.backward()
            # take out gradient dict
            grad_list = []
            for m in self.net.redundant_modules:
                grad_list.append(m.AP_path_alpha.grad.data.clone())
            grad_buffer.append(grad_list)
            reward_buffer.append(reward)
            # unused modules back
            self.net.unused_modules_back()
        # update baseline function
        avg_reward = sum(reward_buffer) / self.arch_search_config.batch_size
        if self.arch_search_config.baseline is None:
            self.arch_search_config.baseline = avg_reward
        else:
            self.arch_search_config.baseline += self.arch_search_config.baseline_decay_weight * \
                                                (avg_reward - self.arch_search_config.baseline)
        # assign gradients
        for idx, m in enumerate(self.net.redundant_modules):
            m.AP_path_alpha.grad.data.zero_()
            for j in range(self.arch_search_config.batch_size):
                m.AP_path_alpha.grad.data += (reward_buffer[j] - self.arch_search_config.baseline) * grad_buffer[j][idx]
            m.AP_path_alpha.grad.data /= self.arch_search_config.batch_size
        # apply gradients
        self.arch_optimizer.step()

        return reward_buffer, net_info_buffer

    def gradient_step(self):
        assert isinstance(self.arch_search_config, GradientArchSearchConfig)
        time0 = time.time()
        if self.arch_search_config.data_batch is None:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = \
                self.run_manager.run_config.train_batch_size
        else:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.arch_search_config.data_batch
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # Mix edge mode
        MixedEdge.MODE = self.arch_search_config.binary_mode
        time1 = time.time()  # time
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
        time2 = time.time()  # time
        # compute output
        self.net.reset_binary_gates()  # random sample binary gates
        self.net.unused_modules_off()  # remove unused module for speedup
        output = self.run_manager.net(images)
        time3 = time.time()  # time
        # loss
        ce_loss = self.run_manager.criterion(output, labels)
        if self.arch_search_config.target_hardware is None:
            expected_value = None
        elif self.arch_search_config.target_hardware == 'mobile':
            expected_value = self.net.expected_latency(self.run_manager.latency_estimator)
        elif self.arch_search_config.target_hardware == 'flops':
            data_shape = [1] + list(self.run_manager.run_config.data_provider.data_shape)
            input_var = torch.zeros(data_shape, device=self.run_manager.device)
            expected_value = self.net.expected_flops(input_var)
        else:
            raise NotImplementedError
        loss = self.arch_search_config.add_regularization_loss(ce_loss, expected_value)
        # compute gradient and do SGD step
        self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()
        # set architecture parameter gradients
        self.net.set_arch_param_grad()
        self.arch_optimizer.step()
        if MixedEdge.MODE == 'two':
            self.net.rescale_updated_arch_param()
        # back to normal mode
        self.net.unused_modules_back()
        MixedEdge.MODE = None
        time4 = time.time()  # time
        if torch.distributed.get_rank() == 0:
            self.write_log(
                '(%.3f, %.3f, %.3f, %.3f)' % (time1 - time0, time2 - time1, time3 - time2, time4 - time3), 'gradient',
                should_print=False, end='\t'
            )
        return loss.data.item(), expected_value.item() if expected_value is not None else None



    def train_stage(self, stage_idx, total_stage_num, fix_net_weights=False):
        if stage_idx == total_stage_num - 1:
            criterion = nn.CrossEntropyLoss().to(self.run_manager.device)
        else:
            criterion = nn.MSELoss().to(self.run_manager.device)

        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch
        
        if self.run_manager.out_log:
            arch_param_num = len(list(self.net.architecture_parameters()))
            binary_gates_num = len(list(self.net.binary_gates()))
            weight_param_num = len(list(self.net.weight_parameters()))
            print(
                '#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
                (arch_param_num, binary_gates_num, weight_param_num)
            )

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        
        avg_epoch_time = AverageMeter()
        avg_batch_time = AverageMeter()

        avg_data_time = AverageMeter()
        avg_prepare_time = AverageMeter()
        avg_teacher_time = AverageMeter()
        avg_sf_time = AverageMeter()
        avg_sb_time = AverageMeter()

        avg_arch_d_time = AverageMeter() 
        avg_arch_t_time = AverageMeter() 
        avg_arch_p_time = AverageMeter() 
        avg_arch_s_time = AverageMeter() 

        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
            if self.run_manager.out_log:
                print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            
            # time log
            batch_time = AverageMeter()

            data_time = AverageMeter()
            prepare_time = AverageMeter()
            teacher_time = AverageMeter()
            sf_time= AverageMeter()
            sb_time= AverageMeter()
            arch_d_time = AverageMeter() 
            arch_t_time = AverageMeter() 
            arch_p_time = AverageMeter() 
            arch_s_time = AverageMeter() 

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()

            # switch to train mode
            self.run_manager.net.train()
            self.teacher_net.train()
            
            epoch_start_time = time.time()
            end = time.time()
            
            for i, (images, labels) in enumerate(data_loader):
                if i > 10:
                    break

                batch_start = end
                images= images.to(self.run_manager.device)
                #torch.cuda.synchronize()
                data_time.update((time.time() - end))
                end = time.time()

                # network entropy
                # net_entropy = self.net.entropy()
                # entropy.update(net_entropy.data.item() / arch_param_num, 1)
                
                x = images
                
                t_num_blocks =  len(self.teacher_net.blocks)
                
                with torch.no_grad():
                    for block_idx in range(t_num_blocks):
                        if block_idx + 1 == t_num_blocks:
                            guide_input = x.detach()
                        x = self.teacher_net(x, block_idx)
            
                    teacher_output = x.detach()
                    in_features = guide_input

                #torch.cuda.synchronize()
                teacher_time.update(time.time() - end)
                end = time.time()
                
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup

                #torch.cuda.synchronize()
                prepare_time.update(time.time() - end)
                end = time.time()

                nas_output = self.run_manager.net(in_features)  # forward (DataParallel)

                #mse_weight = [0.0684, 0.171, 0.3422, 0.2395, 0.5474, 0.3422]
                if stage_idx == total_stage_num - 1:
                    # Use CELoss for last stage
                    guide_loss = criterion(nas_output, teacher_output.softmax(dim=1)) 
                    #labels = labels.to(self.run_manager.device)
                    #acc1, acc5  = accuracy(nas_output, labels, topk=(1,5))
                else:
                    guide_loss = criterion(nas_output, teacher_output)
                
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                guide_loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                
                #torch.cuda.synchronize()
                sb_time.update(time.time()-end)
                end = time.time()
                
                arch_d_time_val = 0
                arch_t_time_val = 0
                arch_p_time_val = 0
                arch_s_time_val = 0
                # skip architecture parameter updates in the first epoch (warmup)
                if epoch >= int((self.run_manager.run_config.n_epochs - 0)/4):
                #if epoch > 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        if j > 0:
                            raise ValueError('do not support multiple steps of architecture params update')
                        if isinstance(self.arch_search_config, RLArchSearchConfig):
                            reward_list, net_info_list = self.rl_update_step(fast=True)
                            used_time = time.time() - start_time
                            log_str = 'REINFORCE [%d-%d]\tTime %.4f\tMean Reward %.4f\t%s' % (
                                epoch + 1, i, used_time, sum(reward_list) / len(reward_list), net_info_list
                            )
                            self.write_log(log_str, prefix='rl', should_print=False)
                        elif isinstance(self.arch_search_config, GradientArchSearchConfig):
                            arch_loss, exp_value, arch_d_time_val, arch_t_time_val, arch_p_time_val, arch_s_time_val = self.gradient_step_stage(stage_idx)
                            if self.run_manager.out_log:
                                log_str = 'Architecture [%d-%d]\t Loss %.4f\t %s %s' % \
                                          (epoch + 1, i, arch_loss,
                                           self.arch_search_config.target_hardware, exp_value)
                                self.write_log(log_str, prefix='gradient', should_print=False)
                        else:
                            raise ValueError('do not support: %s' % type(self.arch_search_config))
                arch_d_time.update(arch_d_time_val)
                arch_t_time.update(arch_t_time_val)
                arch_p_time.update(arch_p_time_val)
                arch_s_time.update(arch_s_time_val)

                # Total batch time
                torch.cuda.synchronize()
                batch_time.update(time.time() - batch_start)

                # Logging
                #reduced_loss = reduce_tensor(guide_loss.data, self.run_manager.world_size)
                #losses.update(reduced_loss.item(), images.size(0))
                #if stage_idx == total_stage_num - 1:
                #    acc1 = reduce_tensor(acc1, self.run_manager.world_size)
                #    acc5 = reduce_tensor(acc5, self.run_manager.world_size)
                #    top1.update(acc1.item(), images.size(0))
                #    top5.update(acc5.item(), images.size(0))

                if self.run_manager.out_log and (i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch):
                                        
                    # training log
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1,top5=top5,lr=lr)
                    self.run_manager.write_log(batch_log, 'train', should_print=True)
                    
                    batch_log = 'Time [{0}][{1}/{2}]\t' \
                                'B {batch_time.val:.4f}({batch_time.avg:.4f})\t' \
                                'D {data_time.val:.4f}({data_time.avg:.4f})\t' \
                                'TF {teacher_time.val:.4f}({teacher_time.avg:.4f})\t' \
                                'P {prepare_time.val:.4f}({prepare_time.avg:.4f})\t' \
                                'SF {sf_time.val:.4f}({sf_time.avg:.4f})\t' \
                                'SB {sb_time.val:.4f}({sb_time.avg:.4f})\t' \
                                'AD {arch_d_time.val:.4f}({arch_d_time.avg:.4f})\t' \
                                'AT {arch_t_time.val:.4f}({arch_t_time.avg:.4f})\t' \
                                'AP {arch_p_time.val:.4f}({arch_p_time.avg:.4f})\t' \
                                'AS {arch_s_time.val:.4f}({arch_s_time.avg:.4f})\t'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time, prepare_time=prepare_time,
                               teacher_time=teacher_time, sf_time=sf_time, sb_time=sb_time,arch_d_time=arch_d_time, arch_t_time= arch_t_time, arch_p_time= arch_p_time, arch_s_time= arch_s_time)

                    self.run_manager.write_log(batch_log, 'time', should_print=False)
                # update time for data_loader
                #torch.cuda.synchronize()
                end = time.time()
            if epoch >= 0:
                avg_epoch_time.update(time.time() - epoch_start_time)
                avg_batch_time.update(batch_time.avg)

                avg_data_time.update(data_time.avg)
                avg_prepare_time.update(prepare_time.avg)
                avg_teacher_time.update(teacher_time.avg)
                avg_sf_time.update(sf_time.avg)
                avg_sb_time.update(sb_time.avg)
                avg_arch_d_time.update(arch_d_time.avg)
                avg_arch_t_time.update(arch_t_time.avg)
                avg_arch_p_time.update(arch_p_time.avg)
                avg_arch_s_time.update(arch_s_time.avg)

                batch_log = 'Epoch [{0}]\t'  \
                            'B {batch_time.val:.4f}({batch_time.avg:.4f})\t' \
                            'D {data_time.val:.4f}({data_time.avg:.4f})\t' \
                            'P {prepare_time.val:.4f}({prepare_time.avg:.4f})\t' \
                            'TF {teacher_time.val:.4f}({teacher_time.avg:.4f})\t' \
                            'SF {sf_time.val:.4f}({sf_time.avg:.4f})\t' \
                            'SB {sb_time.val:.4f}({sb_time.avg:.4f})\t' \
                            'Ad {arch_d_time.val:.4f}({arch_d_time.avg:.4f})\t' \
                            'At {arch_t_time.val:.4f}({arch_t_time.avg:.4f})\t' \
                            'Ap {arch_p_time.val:.4f}({arch_p_time.avg:.4f})\t' \
                            'As {arch_s_time.val:.4f}({arch_s_time.avg:.4f})\t' \
                            'E {epoch_time.val:.4f}({epoch_time.avg:.4f})\t'. \
                    format(epoch + 1, epoch_time= avg_epoch_time, batch_time=avg_batch_time, data_time=avg_data_time, prepare_time=avg_prepare_time,
                           teacher_time=avg_teacher_time, sf_time=avg_sf_time, sb_time=avg_sb_time,
                           arch_d_time= avg_arch_d_time, arch_t_time = avg_arch_t_time,
                           arch_p_time= avg_arch_p_time, arch_s_time = avg_arch_s_time)
                self.run_manager.write_log(batch_log, 'time', should_print=False)

            if self.run_manager.out_log:
                # print current network architecture
                self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
                for stage in self.net.blocks:
                    if hasattr(stage, 'module'):
                        stage = stage.module
                    for idx, block in enumerate(stage):
                        self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
                self.write_log('-' * 60, prefix='arch')

        if self.run_manager.out_log:
            # unwrap DDP
            if self.net.first_conv is not None and hasattr(self.net.first_conv, 'module'):
                self.net.first_conv = self.net.first_conv.module
            if hasattr(self.net.blocks[0], 'module'): 
                self.net.blocks = torch.nn.ModuleList([block.module for block in self.net.blocks])
            if self.net.feature_mix_layer is not None and hasattr(self.net.feature_mix_layer, 'module'):
                self.net.feature_mix_layer = self.net.feature_mix_layer.module
                self.net.classifier = self.net.classifier.module
            # save model
            print('###model saved###')
            self.run_manager.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })


    def gradient_step_stage(self, stage_idx):
        assert isinstance(self.arch_search_config, GradientArchSearchConfig)

        if self.arch_search_config.data_batch is None:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = \
                self.run_manager.run_config.train_batch_size
        else:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.arch_search_config.data_batch
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True

        # switch to train mode
        self.run_manager.net.train()
        self.teacher_net.train()

        # Mix edge mode
        MixedEdge.MODE = self.arch_search_config.binary_mode

        time1 = time.time()  # time
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)

        # Data Load
        #torch.cuda.synchronize() 
        time2 = time.time()
        
        x = images
        t_num_blocks =  len(self.teacher_net.blocks)
        with torch.no_grad():
            for block_idx in range(t_num_blocks):
                if block_idx + 1 == t_num_blocks:
                    guide_input = x.detach()
                x = self.teacher_net(x, block_idx)
        teacher_output = x.detach()
        in_features = guide_input.requires_grad_(True)

        
        # Teacher foward
        #torch.cuda.synchronize()
        time3 = time.time()  
        
        # compute output
        self.net.reset_binary_gates()  # random sample binary gates
        self.net.unused_modules_off()  # remove unused module for speedup
        
        #torch.cuda.synchronize()
        time4 = time.time()  # time

        output = self.run_manager.net(in_features)
        
        # loss
        if stage_idx == self.run_manager.world_size - 1:
            criterion = nn.CrossEntropyLoss().to(self.run_manager.device)
            guide_loss = criterion(output, teacher_output.softmax(dim=1))
        else:
            criterion = nn.MSELoss().to(self.run_manager.device)
            guide_loss  = criterion(output, teacher_output)

        if self.arch_search_config.target_hardware is None:
            expected_value = None
        elif self.arch_search_config.target_hardware == 'mobile':
            expected_value = self.net.expected_latency(self.run_manager.latency_estimator)
        elif self.arch_search_config.target_hardware == 'flops':
            #data_shape = [1] + list(self.run_manager.run_config.data_provider.data_shape)
            data_shape = list(in_features.shape)
            data_shape[0] = 1
            input_var = torch.zeros(data_shape, device=self.run_manager.device)
            expected_value = self.net.expected_flops(input_var)
        else:
            raise NotImplementedError

        #mse_weight = [0.0684, 0.171, 0.3422, 0.2395, 0.5474, 0.3422]
        #loss = self.arch_search_config.add_regularization_loss(guide_loss * mse_weight[stage_idx], expected_value)
        loss = self.arch_search_config.add_regularization_loss(guide_loss, expected_value)
        
        # Student forward

        # compute gradient and do SGD step
        self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()
        # set architecture parameter gradients
        self.net.set_arch_param_grad()
        self.arch_optimizer.step()
        if MixedEdge.MODE == 'two':
            self.net.rescale_updated_arch_param()
        # back to normal mode
        self.net.unused_modules_back()
        MixedEdge.MODE = None
        
        # Student backward
        #torch.cuda.synchronize()
        time5 = time.time()  # time
        if torch.distributed.get_rank() == 0 :
            self.write_log(
                    '(D: %.4f,TF: %.4f,SF: %.4f, SB: %.4f)' % (time2 - time1, time3 - time2, time4 - time3, time5 - time4), 'gradient',
                should_print=False, end='\t'
            )
        return loss.data.item(), expected_value.item() if expected_value is not None else None , (time2- time1), (time3 - time2),(time4-time3), (time5-time4)

    def train_mp(self, my_rank, rank_to_feature_shape, chosen_dp_mp, stage_info, rank_to_batch, rank_to_pgroup, scheme, fix_net_weights=False):
        assert(self.teacher_net is not None)

        batch_size = rank_to_batch[my_rank]
        
        if hasattr(self.run_manager.run_config.train_loader, 'batch_sampler'):
            self.run_manager.run_config.train_loader.batch_sampler.drop_last = True # communication tensor size fixed

        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch

        arch_param_num = len(list(self.net.architecture_parameters()))
        binary_gates_num = len(list(self.net.binary_gates()))
        weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#my_rank: %d\t#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
            (my_rank, arch_param_num, binary_gates_num, weight_param_num)
        )

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        
        avg_epoch_time = AverageMeter()
        avg_batch_time = AverageMeter()

        avg_data_time = AverageMeter()
        avg_teacher_time = AverageMeter()
        avg_prepare_time = AverageMeter()
        avg_sf_time = AverageMeter()
        avg_sb_time = AverageMeter()

        avg_arch_d_time = AverageMeter() 
        avg_arch_t_time = AverageMeter() 
        avg_arch_p_time = AverageMeter() 
        avg_arch_s_time = AverageMeter() 
        avg_arch_send_time = AverageMeter() 


        avg_send_time = AverageMeter()
        
        torch.manual_seed(0)

        #for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
            
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()

            # time log
            batch_time = AverageMeter()
            data_time = AverageMeter()
            teacher_time = AverageMeter()
            prepare_time = AverageMeter()
            sf_time = AverageMeter()
            sb_time = AverageMeter()
            
            arch_d_time = AverageMeter() 
            arch_t_time = AverageMeter() 
            arch_p_time = AverageMeter() 
            arch_s_time = AverageMeter() 
            arch_send_time = AverageMeter() 
            
            # Communication log
            send_time = AverageMeter()

            # switch to train mode
            self.run_manager.net.train()
            self.teacher_net.train() 

            ce_criterion = nn.CrossEntropyLoss().to(self.run_manager.device)
            criterion = nn.MSELoss().to(self.run_manager.device)
            

            if my_rank in range(chosen_dp_mp[0]):
                work_list = data_loader
            else:
                work_list = [ (i,i) for i in range(nBatch)]
                in_buf = torch.empty( rank_to_feature_shape[my_rank], device = self.run_manager.device)

            end = time.time()
            epoch_start_time = end
            r_req_list = []
            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                for i, (images, _) in enumerate(work_list):
                    batch_start = end
                    if my_rank in range(chosen_dp_mp[0]): 
                        in_features = images.to(self.run_manager.device)
                    else:
                        if chosen_dp_mp[0] == 3:
                            if my_rank == 3:
                                for r in range(3):
                                    torch.distributed.recv(tensor = in_buf[int(r * batch_size/ 3): int((r+1)* batch_size / 3)], src = r)
                        elif chosen_dp_mp[0] == 2:
                            if chosen_dp_mp[1] == 2:
                                if my_rank == 2 or my_rank == 3:
                                    torch.distributed.recv(tensor = in_buf, src = my_rank - 2)
                            elif chosen_dp_mp[1] == 1:
                                if my_rank == 2:
                                    torch.distributed.recv(tensor = in_buf[:int(batch_size/2)], src = 0)
                                    torch.distributed.recv(tensor = in_buf[int(batch_size/2):], src = 1)
                                elif my_rank == 3:
                                    torch.distributed.recv(tensor = in_buf, src = 2)
                        elif chosen_dp_mp[0] == 1:
                            if chosen_dp_mp[1] == 3:
                                torch.distributed.recv(tensor = in_buf, src = 0)
                            elif chosen_dp_mp[1] == 2:
                                if my_rank == 1 or my_rank == 2:
                                    torch.distributed.recv(tensor = in_buf, src = 0)
                                elif my_rank == 3:
                                    torch.distributed.recv(tensor = in_buf[:int(batch_size/2)], src = 1)
                                    torch.distributed.recv(tensor = in_buf[int(batch_size/2):], src = 2)
                            elif chosen_dp_mp[1] == 1:
                                if chosen_dp_mp[2] == 2:
                                    if my_rank == 1:
                                        torch.distributed.recv(tensor = in_buf, src = 0)
                                    elif my_rank == 2 or my_rank == 3:
                                        torch.distributed.recv(tensor = in_buf, src = 1)
                                        
                                elif chosen_dp_mp[2] == 1:
                                    torch.distributed.recv(tensor = in_buf, src = my_rank - 1)

                        #print(my_rank,':', in_buf.shape)
                        in_features = in_buf

                    x = in_features
                        
                    #torch.cuda.synchronize()
                    data_time.update(time.time() - end)
                    end = time.time()

                    t_num_blocks =  len(self.teacher_net.blocks)

                    guide_features = []
                    
                    with torch.no_grad():
                        for block_idx in range(t_num_blocks):
                            #print(my_rank, block_idx)
                            guide_features.append(x.detach()) 
                            x = self.teacher_net(x, block_idx)
                
                        guide_features.append(x.detach())
                    #print(my_rank, [ f.shape for f in guide_features])
                    
                    #torch.cuda.synchronize()
                    teacher_time.update(time.time() - end)
                    end = time.time()
                    
                    s_req_list = []
                    if self.run_manager.world_size == 4:
                        # When world size is 4
                        if chosen_dp_mp[0] == 3:
                            if my_rank in range(3):
                                s_req_list.append(torch.distributed.isend(tensor = guide_features[-1], dst = 3))
                        elif chosen_dp_mp[0] == 2:
                            if chosen_dp_mp[1] == 2:
                                if my_rank in range(2):
                                    s_req_list.append(torch.distributed.isend(tensor = guide_features[-1] , dst = my_rank + 2))
                            elif chosen_dp_mp[1] == 1:
                                if my_rank in range(2):
                                    s_req_list.append(torch.distributed.isend(tensor = guide_features[-1] , dst = 2))
                                elif my_rank == 2:
                                    s_req_list.append(torch.distributed.isend(tensor =guide_features[-1] , dst = 3))

                        elif chosen_dp_mp[0] == 1:
                            if chosen_dp_mp[1] == 3:
                                if my_rank == 0:
                                    for target in range(3):
                                        s_req_list.append(torch.distributed.isend(tensor =guide_features[-1][int(target * batch_size/3):int((target+1) * batch_size/3)], dst = (target+1)))
                            elif chosen_dp_mp[1] == 2:
                                if my_rank == 0:
                                    for target in range(2):
                                        s_req_list.append(torch.distributed.isend(tensor =guide_features[-1][int(target * batch_size/2):int((target+1) * batch_size/2)], dst = (target+1)))
                                elif my_rank in range(1,3):
                                    s_req_list.append(torch.distributed.isend(tensor =guide_features[-1] , dst = 3))

                            elif chosen_dp_mp[1] == 1:
                                if my_rank == 0 :
                                    s_req_list.append(torch.distributed.isend(tensor = guide_features[-1], dst = 1))
                                else: 
                                    if chosen_dp_mp[2] == 2:
                                        if my_rank == 1:
                                            for target in range(2):
                                                s_req_list.append(torch.distributed.isend(tensor = guide_features[-1][int(target * batch_size/2):int((target+1) * batch_size/2)], dst = (target+2)))
                                    elif chosen_dp_mp[1] == 1:
                                        if my_rank != 3:
                                            s_req_list.append(torch.distributed.isend(tensor =guide_features[-1] , dst = (my_rank+1)))

                    send_time.update(time.time() - end)
                    end = time.time()

                                            
                    # lr
                    lr = self.run_manager.run_config.adjust_learning_rate(
                        self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                    )
                    # network entropy
                    # net_entropy = self.net.entropy()
                    # entropy.update(net_entropy.data.item() / arch_param_num, 1)

                    self.net.reset_binary_gates()  # random sample binary gates
                    self.net.unused_modules_off()  # remove unused module for speedup

                    #print('[%d]'%my_rank, len(list(self.net.weight_parameters())))
                    #torch.cuda.synchronize()
                    prepare_time.update(time.time() - end)
                    end = time.time()

                    self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                    
                    for block_idx in range(len(self.net.blocks)):
                        nas_output = self.net(guide_features[block_idx], block_idx)
                        #print(my_rank,'{', block_idx,':',nas_output.shape, '}')
                        if block_idx == (len(self.net.blocks) - 1) and stage_info[my_rank][1] == 5:
                            loss = ce_criterion(nas_output, guide_features[block_idx+1].softmax(dim=1))
                        else:
                            loss = criterion(nas_output, guide_features[block_idx+1])
                        
                        # compute gradient and do SGD step
                        loss.backward()
                    

                    
                    #if rank_to_pgroup[my_rank] is not None:
                    #    grad = list() 
                    #    for param in self.net.weight_parameters():
                    #        if (param is not None) and (param.grad is not None) and (len(param.size()) > 0):
                    #            grad.append(param.grad.data) 
                    #    buf =TensorBuffer(grad)
                    #    #print(buf.buffer.element_size() * buf.buffer.nelement())     
                    #    torch.distributed.all_reduce(buf.buffer, group = rank_to_pgroup[my_rank])
                    #    buf.buffer /= torch.distributed.get_world_size(group = rank_to_pgroup[my_rank])
                    #    buf.unpack(grad)
                    #
                    #    cnt = 0
                    #    for param in self.net.weight_parameters():
                    #        if (param is not None) and (param.grad is not None) and (len(param.size()) > 0):
                    #            param.grad.data = grad[cnt]
                    #            cnt += 1

                    #teacher relaying 
                    if scheme == 'tr':
                        torch.distributed.barrier() 

                    self.run_manager.optimizer.step()  # update weight parameters
                    
                    # unused modules back
                    self.net.unused_modules_back()

                    #torch.cuda.synchronize() 
                    sb_time.update(time.time() - end)
                    end = time.time()
    

                    for s_req in s_req_list:
                        s_req.wait()
                    
                                    
                    arch_d_time_val = 0
                    arch_t_time_val = 0
                    arch_p_time_val = 0
                    arch_s_time_val = 0
                    arch_send_time_val = 0

                    # skip architecture parameter updates in the first epoch
                    #if epoch > 0:
                    if epoch >= int((self.run_manager.run_config.n_epochs)/4):
                        # update architecture parameters according to update_schedule
                        for j in range(update_schedule.get(i, 0)):
                            if isinstance(self.arch_search_config, GradientArchSearchConfig):
                                arch_loss, exp_value, arch_d_time_val, arch_t_time_val,arch_send_time_val, arch_p_time_val, arch_s_time_val = self.guided_gradient_step(my_rank, rank_to_feature_shape, chosen_dp_mp, batch_size, in_features, stage_info, rank_to_pgroup, scheme)
                                log_str = 'Architecture [%d-%d]\t Loss %.4f\t %s %s' % \
                                        (epoch + 1, i, arch_loss, self.arch_search_config.target_hardware, exp_value)
                                self.write_log(log_str, prefix='gradient', should_print=False)
                            else:
                                raise ValueError('do not support: %s' % type(self.arch_search_config))

                    arch_d_time.update( arch_d_time_val )
                    arch_t_time.update( arch_t_time_val )
                    arch_p_time.update( arch_p_time_val )
                    arch_s_time.update( arch_s_time_val )
                    arch_send_time.update( arch_send_time_val )
                
                    torch.cuda.synchronize()
                    batch_time.update(time.time() - batch_start)

                    # Logging
                    losses.update(loss, in_features.size(0))

                    # training log
                    if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                        batch_log = 'Train [{0}][{1}/{2}]\t' \
                                    'B {batch_time.val:.4f}({batch_time.avg:.4f})\t' \
                                    'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                    'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                    'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                    'Teacher acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                            format(epoch + 1, i, nBatch - 1, batch_time = batch_time,
                                losses=losses, entropy=entropy, top1=top1,top5=top5,lr=lr)
                        self.run_manager.write_log(batch_log, 'train')
                        
                        batch_log = 'Time [{0}][{1}/{2}]\t' \
                                    'B {batch_time.val:.4f}({batch_time.avg:.4f})\t' \
                                    'D {data_time.val:.4f}({data_time.avg:.4f})\t' \
                                    'TF {teacher_time.val:.4f}({teacher_time.avg:.4f})\t' \
                                    'P {prepare_time.val:.4f}({prepare_time.avg:.4f})\t' \
                                    'SF {sf_time.val:.4f}({sf_time.avg:.4f})\t' \
                                    'SB {sb_time.val:.4f}({sb_time.avg:.4f})\t' \
                                    'send {send_time.val:.4f}({send_time.avg:.4f})\t'. \
                            format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                                teacher_time=teacher_time, prepare_time=prepare_time, 
                                sf_time=sf_time, sb_time=sb_time, send_time= send_time)

                        #batch_log = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                        self.run_manager.write_log(batch_log, 'time', should_print=False)
                    """
                    # Do not use first batch time
                    if i == 1: 
                        batch_time = AverageMeter()
                        data_time = AverageMeter()
                        prepare_time = AverageMeter()
                        teacher_time = AverageMeter()
                        sf_time= AverageMeter()
                        sb_time= AverageMeter()
                    """
                    # update time for data_loader
                    end = time.time()
                    if my_rank == 0:
                        prof.step()

            if epoch >= 0:
                avg_epoch_time.update(time.time() - epoch_start_time)
                avg_batch_time.update(batch_time.avg)

                avg_data_time.update(data_time.avg)
                avg_teacher_time.update(teacher_time.avg)
                avg_prepare_time.update(prepare_time.avg)
                avg_sf_time.update(sf_time.avg)
                avg_sb_time.update(sb_time.avg)

                avg_arch_d_time.update(arch_d_time.avg)
                avg_arch_t_time.update(arch_t_time.avg)
                avg_arch_p_time.update(arch_p_time.avg)
                avg_arch_s_time.update(arch_s_time.avg)
                avg_arch_send_time.update(arch_send_time.avg)

                avg_send_time.update(send_time.avg)

                batch_log = 'Rank [{0}] at [{1:>3}]\t' \
                            'B {batch_time.val:.4f}({batch_time.avg:.4f})\t' \
                            'D {data_time.val:.4f}({data_time.avg:.4f})\t' \
                            'T {teacher_time.val:.4f}({teacher_time.avg:.4f})\t' \
                            'P {prepare_time.val:.4f}({prepare_time.avg:.4f})\t' \
                            'SF {sf_time.val:.4f}({sf_time.avg:.4f})\t' \
                            'SB {sb_time.val:.4f}({sb_time.avg:.4f})\t' \
                            'AD {arch_d_time.val:.4f}({arch_d_time.avg:.4f})\t' \
                            'AT {arch_t_time.val:.4f}({arch_t_time.avg:.4f})\t' \
                            'AP {arch_p_time.val:.4f}({arch_p_time.avg:.4f})\t' \
                            'AS {arch_s_time.val:.4f}({arch_s_time.avg:.4f})\t' \
                            'ASend {arch_send_time.val:.4f}({arch_send_time.avg:.4f})\t' \
                            'send {send_time.val:.4f}({send_time.avg:.4f})\t' \
                            'Epoch {epoch_time.val:.4f}({epoch_time.avg:.4f})\t'. \
                    format(my_rank, epoch + 1, epoch_time= avg_epoch_time, batch_time=avg_batch_time, data_time=avg_data_time,
                           teacher_time=avg_teacher_time, prepare_time=avg_prepare_time, sf_time=avg_sf_time, sb_time=avg_sb_time,
                           arch_d_time =avg_arch_d_time,
                           arch_t_time =avg_arch_t_time,
                           arch_p_time =avg_arch_p_time,
                           arch_s_time =avg_arch_s_time,
                           arch_send_time =avg_arch_send_time,
                           send_time=avg_send_time)
                self.run_manager.write_log(batch_log, 'time', should_print=False)

            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            for stage in self.net.blocks:
                if hasattr(stage, 'module'):
                    stage = stage.module
                for idx, block in enumerate(stage):
                    self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

        # save model with unwarped DDP
        if self.net.first_conv is not None:
            self.net.first_conv = self.net.first_conv.module
        self.net.blocks = torch.nn.ModuleList([block.module for block in self.net.blocks])
        if self.net.feature_mix_layer is not None:
            self.net.feature_mix_layer = self.net.feature_mix_layer.module
            self.net.classifier = self.net.classifier.module
        self.run_manager.save_model({
            'warmup': False,
            'epoch': epoch,
            'weight_optimizer': self.run_manager.optimizer.state_dict(),
            'arch_optimizer': self.arch_optimizer.state_dict(),
            'state_dict': self.net.state_dict()
        })


    def guided_gradient_step(self, my_rank, rank_to_feature_shape, chosen_dp_mp, batch_size, in_buf, stage_info, rank_to_pgroup, scheme):
        assert isinstance(self.arch_search_config, GradientArchSearchConfig)
        
        ce_criterion = nn.CrossEntropyLoss().to(self.run_manager.device)
        criterion = nn.MSELoss().to(self.run_manager.device)


        if hasattr(self.run_manager.run_config.valid_loader, 'batch_sampler'):
            if self.arch_search_config.data_batch is None:
                self.run_manager.run_config.valid_loader.batch_sampler.batch_size = \
                self.run_manager.run_config.train_batch_size
            else:
                self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.arch_search_config.data_batch

            self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True

        self.run_manager.net.train()
        self.teacher_net.train()

        
        # Mix edge mode
        MixedEdge.MODE = self.arch_search_config.binary_mode
        

        time1 = time.time()  # time

        if my_rank in range(chosen_dp_mp[0]): 
            images, _ = self.run_manager.run_config.valid_next_batch
            # Distribute data to each worker
            #images = images[ int(batch_size/chosen_dp_mp[0] * my_rank) : int(batch_size/chosen_dp_mp[0] * (my_rank + 1))].to(self.run_manager.device)
            #print(my_rank,':', images.shape)
            in_features = images.to(self.run_manager.device)
        else:
            if chosen_dp_mp[0] == 3:
                if my_rank == 3:
                    for r in range(3):
                        torch.distributed.recv(tensor = in_buf[int(r * batch_size/ 3): int((r+1)* batch_size / 3)], src = r)
            elif chosen_dp_mp[0] == 2:
                if chosen_dp_mp[1] == 2:
                    if my_rank == 2 or my_rank == 3:
                        torch.distributed.recv(tensor = in_buf, src = my_rank - 2)
                elif chosen_dp_mp[1] == 1:
                    if my_rank == 2:
                        torch.distributed.recv(tensor = in_buf[:int(batch_size/2)], src = 0)
                        torch.distributed.recv(tensor = in_buf[int(batch_size/2):], src = 1)
                    elif my_rank == 3:
                        torch.distributed.recv(tensor = in_buf, src = 2)
            elif chosen_dp_mp[0] == 1:
                if chosen_dp_mp[1] == 3:
                    torch.distributed.recv(tensor = in_buf, src = 0)
                elif chosen_dp_mp[1] == 2:
                    if my_rank == 1 or my_rank == 2:
                        torch.distributed.recv(tensor = in_buf, src = 0)
                    elif my_rank == 3:
                        torch.distributed.recv(tensor = in_buf[:int(batch_size/2)], src = 1)
                        torch.distributed.recv(tensor = in_buf[int(batch_size/2):], src = 2)
                elif chosen_dp_mp[1] == 1:
                    if chosen_dp_mp[2] == 2:
                        if my_rank == 1:
                            torch.distributed.recv(tensor = in_buf, src = 0)
                        elif my_rank == 2 or my_rank == 3:
                            torch.distributed.recv(tensor = in_buf, src = 1)
                            
                    elif chosen_dp_mp[2] == 1:
                        torch.distributed.recv(tensor = in_buf, src = my_rank - 1)

            #print(my_rank,':', in_buf.shape)
            in_features = in_buf

        x = in_features
                  
        # sample a batch of data from validation set
        # Data load
        #torch.cuda.synchronize()
        time2 = time.time()  # time

        
        t_num_blocks =  len(self.teacher_net.blocks)

        guide_features = []
        
        with torch.no_grad():
            for block_idx in range(t_num_blocks):
                #print(my_rank, block_idx)
                guide_features.append(x.detach().requires_grad_(True))
                x = self.teacher_net(x, block_idx)
    
            guide_features.append(x.detach().requires_grad_(True))
        
        # Teacher
        #torch.cuda.synchronize() 
        time3 = time.time()

        s_req_list = []
        if self.run_manager.world_size == 4:
            # When world size is 4
            if chosen_dp_mp[0] == 3:
                if my_rank in range(3):
                    s_req_list.append(torch.distributed.isend(tensor = x, dst = 3))
            elif chosen_dp_mp[0] == 2:
                if chosen_dp_mp[1] == 2:
                    if my_rank in range(2):
                        s_req_list.append(torch.distributed.isend(tensor = x, dst = my_rank + 2))
                elif chosen_dp_mp[1] == 1:
                    if my_rank in range(2):
                        s_req_list.append(torch.distributed.isend(tensor = x, dst = 2))
                    elif my_rank == 2:
                        s_req_list.append(torch.distributed.isend(tensor = x, dst = 3))

            elif chosen_dp_mp[0] == 1:
                if chosen_dp_mp[1] == 3:
                    if my_rank == 0:
                        for target in range(3):
                            s_req_list.append(torch.distributed.isend(tensor =x[int(target * batch_size/3):int((target+1) * batch_size/3)], dst = (target+1)))
                elif chosen_dp_mp[1] == 2:
                    if my_rank == 0:
                        for target in range(2):
                            s_req_list.append(torch.distributed.isend(tensor = x[int(target * batch_size/2):int((target+1) * batch_size/2)], dst = (target+1)))
                    elif my_rank in range(1,3):
                        s_req_list.append(torch.distributed.isend(tensor = x, dst = 3))

                elif chosen_dp_mp[1] == 1:
                    if my_rank == 0 :
                        s_req_list.append(torch.distributed.isend(tensor = x, dst = 1))
                    else: 
                        if chosen_dp_mp[2] == 2:
                            if my_rank == 1:
                                for target in range(2):
                                    s_req_list.append(torch.distributed.isend(tensor = x[int(target * batch_size/2):int((target+1) * batch_size/2)], dst = (target+2)))
                        elif chosen_dp_mp[1] == 1:
                            if my_rank != 3:
                                s_req_list.append(torch.distributed.isend(tensor = x, dst = (my_rank+1)))
        # send time 
        time4 = time.time()

        # compute output
        self.net.reset_binary_gates()  # random sample binary gates
        self.net.unused_modules_off()  # remove unused module for speedup
        
        #torch.cuda.synchronize() 
        time5 = time.time()

        self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        for block_idx in range(len(self.net.blocks)):
            nas_output = self.net(guide_features[block_idx], block_idx)
            #print(my_rank,'{', block_idx,':',nas_output.shape, '}')
            if block_idx == (len(self.net.blocks) - 1) and stage_info[my_rank][1] == 5:
                guide_loss = ce_criterion(nas_output, guide_features[block_idx+1].softmax(dim=1))
            else:
                guide_loss = criterion(nas_output, guide_features[block_idx+1])
            
            if self.arch_search_config.target_hardware is None:
                expected_value = None
            elif self.arch_search_config.target_hardware == 'mobile':
                expected_value = self.net.expected_latency(self.run_manager.latency_estimator)
            elif self.arch_search_config.target_hardware == 'flops':
                #data_shape = [1] + list(self.run_manager.run_config.data_provider.data_shape)
                data_shape = list(guide_features[block_idx].shape)
                data_shape[0] = 1
                input_var = torch.zeros(data_shape, device=self.run_manager.device)
                expected_value = self.net.expected_flops(input_var, block_idx)
            else:
                raise NotImplementedError

            loss = self.arch_search_config.add_regularization_loss(guide_loss, expected_value, block_idx = block_idx)
            
            self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
            # compute gradient and do SGD step
            loss.backward()
       
        # teacher relaying 
        if scheme == 'tr':
            torch.distributed.barrier()

        self.net.set_arch_param_grad()

        self.arch_optimizer.step()
        
        if MixedEdge.MODE == 'two':
            self.net.rescale_updated_arch_param()
        
        # back to normal mode
        self.net.unused_modules_back()

        #if rank_to_pgroup[my_rank] is not None:
        #    grad = list() 
        #    for param in self.net.architecture_parameters():
        #        if (param is not None) and (param.grad is not None) and (len(param.size()) > 0):
        #            grad.append(param.grad.data)
        #    buf =TensorBuffer(grad)
        #    torch.distributed.all_reduce(buf.buffer, group = rank_to_pgroup[my_rank])
        #    
        #    buf.buffer /= torch.distributed.get_world_size(group = rank_to_pgroup[my_rank])
        #    buf.unpack(grad)
        # 
        #    cnt = 0
        #    for param in self.net.architecture_parameters():
        #        if (param is not None) and (param.grad is not None) and (len(param.size()) > 0):
        #            param.grad.data = grad[cnt]
        #            cnt += 1


        
                
        # Student backward 
        #torch.cuda.synchronize() 
        time6= time.time()  # time
                    
        for s_req in s_req_list:
            s_req.wait()
        
        MixedEdge.MODE = None
        self.write_log(
                '(D:%.4f, TF: %.4f, P : %.4f, S: %.4f, send: %.4f)' % 
                (time2 - time1, time3 - time2, time4 - time3, time5 - time4, time6 - time5), 'gradient',
            should_print=False, end='\t'
        )
        return loss.data.item(), expected_value.item() if expected_value is not None else None, (time2 - time1), (time3-time2), (time4 - time3), (time5 - time4), (time6-time5)

