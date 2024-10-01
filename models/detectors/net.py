from models.backbones.pvtv2 import pvt_v2_b4
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SpatialCoordAttention(nn.Module):
    def __init__(self,in_channels, reduction=16, bias=False,kernel_size=3,act=nn.ReLU()):
        super(SpatialCoordAttention, self).__init__()

        assert kernel_size in (3,7),'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.global_att = nn.Conv2d(2, 1, kernel_size=kernel_size,padding= padding, bias=bias)

        # self.local_att = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=bias),  # fc1
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=bias),  # fc2
        #     # nn.Sigmoid()
        # )

        self.local_h = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels*2 // reduction, kernel_size=kernel_size,padding= padding, bias=bias),  # fc1
            act,
            nn.Conv1d(in_channels*2 // reduction, in_channels, kernel_size=kernel_size,padding= padding, bias=bias),  # fc2
            nn.Sigmoid()
        )
        self.local_w = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels*2 // reduction, kernel_size=kernel_size,padding= padding, bias=bias),  # fc1
            act,
            nn.Conv1d(in_channels*2 // reduction, in_channels, kernel_size=kernel_size,padding= padding, bias=bias),  # fc2
            nn.Sigmoid()
        )


    def forward(self, x):
        # SA
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        global_out = torch.cat([avg_out, max_out], dim=1)
        global_out = self.global_att(global_out)
        # HA
        avg_h = torch.mean(x, dim=3, keepdim=True).squeeze(3)
        max_h = torch.max(x, dim=3, keepdim=True).values.squeeze(3)
        local_h = torch.cat([avg_h,max_h],dim=1)
        local_h = self.local_h(local_h).unsqueeze(3)
        # WA
        avg_w = torch.mean(x, dim=2, keepdim=True).squeeze(2)
        max_w = torch.max(x, dim=2, keepdim=True).values.squeeze(2)
        local_w = torch.cat([avg_w,max_w],dim=1)
        local_w = self.local_w(local_w).unsqueeze(2)

        global_out = torch.sigmoid(global_out)
        local_out = local_h.expand_as(x) * local_w.expand_as(x)
        return global_out+local_out

# CAM中的Conv2d换为Linear即为SE注意力机制
# 注意力论文：CAM：如果只用一个self.fc,即avg和max共用一个Shared MLP，即结构为CAM：Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, bias=False,act=nn.ReLU()):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=bias),# fc1
            # nn.ReLU(inplace=True),
            act,
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=bias),# fc2
            # nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=bias),  # fc1
            # nn.ReLU(inplace=True),
            act,
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=bias),  # fc2
            # nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc2(self.max_pool(x))
        out = nn.Sigmoid()(avg_out+max_out)
        return out

########################################################################################
class SCCAttention_Parallel(nn.Module):
    def __init__(self, in_channels, bias=False,reduction=16,act=nn.ReLU()):
        super(SCCAttention_Parallel, self).__init__()
        self.spatial_att = SpatialCoordAttention(in_channels, reduction,bias=bias,act=act)
        self.channel_att = ChannelAttention(in_channels, reduction,bias=bias,act=act)

    def forward(self, x):
        spa_att = self.spatial_att(x)
        cha_att = self.channel_att(x)
        x_spa = spa_att * x
        x_cha = cha_att * x
        out = x_spa * x_cha
        return out

##########################################################################

class InformationExtractor(nn.Module):
    def __init__(self, in_channels, kernel_size, reduction, bias, act):
        super(InformationExtractor, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,padding=(kernel_size//2),stride=1, bias=bias))
        modules_body.append(act)
        modules_body.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,padding=(kernel_size//2),stride=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

        self.sccAttentionParallel = SCCAttention_Parallel(in_channels,reduction=reduction,bias=bias,act=act)


    def forward(self, x):
        res = self.body(x)
        res = self.sccAttentionParallel(res)
        res += x
        return res

class CISM(nn.Module):
    def __init__(self, in_channels_sementic,in_channels_edge,bias=False,reduction=16,act=nn.ReLU()):
        super(CISM, self).__init__()

        # act_fn = nn.ReLU(inplace=True)

        self.layer_sementic = BasicConv2d(in_channels_sementic, in_channels_sementic, kernel_size=3, stride=1, padding=1,bias=bias)
        self.layer_edge = BasicConv2d(in_channels_edge, in_channels_edge, kernel_size=3, stride=1, padding=1,bias=bias)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels_sementic+in_channels_edge, (in_channels_sementic+in_channels_edge) // reduction, kernel_size=3,padding=1, bias=bias),  # fc1
            # nn.ReLU(inplace=True),
            act,
            nn.Conv2d((in_channels_sementic+in_channels_edge) // reduction, in_channels_sementic+in_channels_edge, kernel_size=3,padding=1, bias=bias),  # fc2
            # nn.Sigmoid()
        )

        self.gate_edge = BasicConv2d(in_channels_sementic+in_channels_edge, 1, kernel_size=3,stride=1,padding=1, bias=bias)

        # self.same_channels_with_edge = nn.Conv2d(in_channels_sementic, in_channels_edge, kernel_size=1, bias=True)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x_sementic, x_edge):
        ###
        x_sementic = self.layer_sementic(x_sementic)
        x_edge = self.layer_edge(x_edge)

        cat_fea = torch.cat([x_sementic, x_edge], dim=1)
        att = nn.Sigmoid()(self.fc(cat_fea))
        att = self.max_pool(att)
        cat_fea = cat_fea*att

        ###
        att_edge_feature = self.gate_edge(cat_fea)

        att_edge = nn.Sigmoid()(att_edge_feature)

        x_sementic = x_sementic * (1 - att_edge)
        x_edge = x_edge * att_edge

        x_fusion = x_sementic + x_edge

        return x_fusion


class VSSNet(nn.Module):
    def __init__(self, in_channels=128, reduction=4, bias=False,
                 act=nn.ReLU(inplace=True)):
        super(VSSNet, self).__init__()

        self.backbone = pvt_v2_b4()  # [64, 128, 320, 512]
        path = './pretrained/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)


        self.cism = CISM(in_channels_sementic=in_channels,in_channels_edge=in_channels,bias=bias,reduction=reduction,act=act)

        self.decoder_level5 = [InformationExtractor(in_channels, kernel_size=3, reduction=reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level4 = [InformationExtractor(in_channels*2, kernel_size=3, reduction=reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level3 = [InformationExtractor(in_channels*3, kernel_size=3, reduction=reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level2_1 = [InformationExtractor(in_channels*4, kernel_size=3, reduction=reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level2_2 = [InformationExtractor(in_channels, kernel_size=3, reduction=reduction, bias=bias, act=act) for _ in range(2)]


        self.decoder_level5 = nn.Sequential(*self.decoder_level5)

        self.decoder_level4 = nn.Sequential(*self.decoder_level4)

        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.decoder_level2_1 = nn.Sequential(*self.decoder_level2_1)

        self.decoder_level2_2= nn.Sequential(*self.decoder_level2_2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out_IE_level5 = nn.Conv2d(in_channels, 1, 1)
        self.out_IE_level4 = nn.Conv2d(in_channels, 1, 1)
        self.out_IE_level3 = nn.Conv2d(in_channels, 1, 1)
        self.out_IE_level2 = nn.Conv2d(in_channels, 1, 1)

        self.out_CISM = nn.Conv2d(in_channels, 1, 1)

        # Smooth layers
        self.smooth5 = BasicConv2d(1*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth4 = BasicConv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3 = BasicConv2d(3*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = BasicConv2d(4 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = BasicConv2d(1*in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Top layer
        self.toplayer = BasicConv2d(512, in_channels, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer4 = BasicConv2d(320, in_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = BasicConv2d(128, in_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer2_1 = BasicConv2d(64, in_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer2_2 = BasicConv2d(64, in_channels, kernel_size=1, stride=1, padding=0)

        # 以下属性未使用
        # self.compress_out = BasicConv2d(2 * in_channels, in_channels, kernel_size=8, stride=4, padding=2)
        # self.compress_out2 = BasicConv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x2 = pvt[0]
        x3 = pvt[1]
        x4 = pvt[2]
        x5 = pvt[3]

        #  top-down
        x5_t = self.toplayer(x5)
        x5_t_feed = self.decoder_level5(x5_t)

        x4_t = self.latlayer4(x4)
        x4_t_feed = torch.cat((x4_t,self.upsample(x5_t_feed)),1)
        x4_t_feed = self.decoder_level4(x4_t_feed)

        x3_t = self.latlayer3(x3)
        x3_t_feed = torch.cat((x3_t, self.upsample(x4_t_feed)), 1)
        x3_t_feed = self.decoder_level3(x3_t_feed)

        x2_t = self.latlayer2_1(x2)
        x2_t_feed = torch.cat((x2_t, self.upsample(x3_t_feed)), 1)
        x2_t_feed = self.decoder_level2_1(x2_t_feed)


        stage_loss = list()
        #Smooth+predict
        semantic_feature = self.smooth5(x5_t_feed)
        prediction = self.out_IE_level5(semantic_feature)
        prediction = F.interpolate(prediction,scale_factor=32,mode='bilinear')
        stage_loss.append(prediction)

        semantic_feature = self.smooth4(x4_t_feed)
        prediction = self.out_IE_level4(semantic_feature)
        prediction = F.interpolate(prediction, scale_factor=16, mode='bilinear')
        stage_loss.append(prediction)

        semantic_feature = self.smooth3(x3_t_feed)
        prediction = self.out_IE_level3(semantic_feature)
        prediction = F.interpolate(prediction, scale_factor=8, mode='bilinear')
        stage_loss.append(prediction)

        semantic_feature = self.smooth2_1(x2_t_feed)
        prediction = self.out_IE_level2(semantic_feature)
        prediction = F.interpolate(prediction, scale_factor=4, mode='bilinear')
        stage_loss.append(prediction)


        vision_feature = self.decoder_level2_2(self.latlayer2_2(x2))# TODO:与09的唯一差别（顺序），不过感觉我这个更合理，解决方案可以再接一个smooth
        vision_feature = self.smooth2_2(vision_feature)

        select_feature = self.cism(semantic_feature, vision_feature)

        prediction = self.out_CISM(select_feature)
        prediction = F.interpolate(prediction, scale_factor=4, mode='bilinear')
        return stage_loss, prediction

#
if __name__ == '__main__':
    model = VSSNet(in_channels=128).cuda()
    input_tensor = torch.randn(1, 3, 384, 384).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
