# Quantitative research question
## Do DAO members share common interests outside of the DAO?
- For now: Did any of the top X addresses that have voted in DAOs also transact with any of the top X NFT projects' smart contracts?

## Do DAO members that show cultural assortativity share more DAOs affiliations?
- Cultural assortativity
    - Shared out-of-DAO transactions with e.g. NFT smart contracts
    - Token holdings?
- Shared DAO memberships: Based on voting in the same DAOs

Not feasible to address right now, because we will focus on the top X, which is too small of a potential overlap space

# Research strategy

## Data

- Snapshot: Download voters in DAOs
    - Address of voter
    - Vote
        - Datetime
        - DAO
- OpenSea: Get addresses of NFT smart contracts
    - API
        - https://docs.opensea.io/reference/retrieving-collections
- Etherscan: Get transactions of voters

# Potential points of failure

- Transactions on OpenSea include bidding on NFTs (no transfer of the NFT yet)
    - Filter these out
- Some NFT collections on OpenSea do not have smart contracts?
    - Example without: https://opensea.io/collection/the-pavone
    - Example with: https://opensea.io/collection/cavemanlabs
        - https://etherscan.io/address/0x7018ec2181bb1c6e52adf0b4e8d679efecf2b8d1


# Preliminary data analysis and visualization