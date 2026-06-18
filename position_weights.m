function sw = position_weights(nn)
    sw = [1 nn(1)*nn(2)]; % position of weights and biasses
    sw = [sw; (sw(1,2)+1) (sw(1,2)+nn(2))];
    sw = [sw; (sw(2,2)+1) (sw(2,2)+(nn(2)+1)*nn(3))];
    sw = [sw; (sw(3,2)+1) (sw(3,2)+nn(3))];
    sw = [sw; (sw(4,2)+1) (sw(4,2)+nn(3)*nn(4))];
    sw = [sw; (sw(5,2)+1) (sw(5,2)+1)];
end