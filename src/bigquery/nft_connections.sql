SELECT token_address,from_address,to_address,value FROM `bigquery-public-data.crypto_ethereum.token_transfers` WHERE DATE(block_timestamp) = "2022-10-11" and transaction_hash = '0x12e358ffdbcc7ed504e07bd98493df2f184a151ec3c8c3ef2fa7f5227846c83d'

-- Results:
-- token_address: 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb
-- from_address: 0x1c41d8667e815451c6efc242367cd15f0a4564dd
-- to_address: 0x163e10cccfbc559c7d28eecf0e27ef10888f6607
-- value: 1

-- Compared to etherscan: https://etherscan.io/tx/0x12e358ffdbcc7ed504e07bd98493df2f184a151ec3c8c3ef2fa7f5227846c83d
-- token_address = interacted with (to)


