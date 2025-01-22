module ImmGen#(parameter Width = 32) (
    input [Width-1:0] inst,
    output reg signed [Width-1:0] imm
);
    // ImmGen generate imm value based on opcode

    wire [6:0] opcode = inst[6:0];
    always @(*) 
    begin
        case(opcode)
            7'b0010011: imm = {{20{inst[31]}}, inst[31:20]}; // I-type
            7'b0100011: imm = {{20{inst[31]}}, inst[31:25], inst[11:7]}; // S-type
            7'b1100011: imm = {{20{inst[31]}}, inst[7], inst[30:25], inst[11:8], 1'b0}; // B-type
            default: imm = 32'b0; // Default case
	   endcase
    end
            
endmodule

