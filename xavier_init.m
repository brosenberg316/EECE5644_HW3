function init_vals = xavier_init(num_rows,num_cols)
% Xavier initialization as described in "Understanding the difficulty of training deep feedforward neural networks"
% http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
Xavier = (sqrt(6)/sqrt(num_rows+num_cols));
init_vals = -Xavier + (2*Xavier)*rand(num_rows,num_cols);