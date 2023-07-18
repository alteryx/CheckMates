Release Notes
-------------
**Future Releases**
    * Enhancements
        * Added in github workflows that are relevant to `DataChecks`, from `EvalML` repository, and modified to fit `DataChecks` wherever possible :pr:`2`
        * Implemented linters and have them successfully running :pr:`2`
    * Fixes
        * Workflows are now up and running properly :pr:`2`
    * Changes
        * Updated comments to reflect `DataChecks` repository :pr:`2`
        * Restructured file directory to categorize data checks between `datacheck_meta` and `checks` :pr:`2`
        * Restructured pdm packaging to only be relevant to `DataChecks`, now to be renamed to `CheckMate` :pr:`2`
    * Documentation Changes
    * Testing Changes
        * Removed integration testing due to irrelevance with `datacheck_meta` and `checks` :pr:`2`

**v0.1.2 July 6, 2023**

* Enhancements
    * Transfer over base `Data Checks` and `IDColumnData Checks` from the `EvalML` repo :pr:`1`
* Fixes
    * Fixed pytest failures :pr:`1`
* Changes
    * Removed all `EvalML` dependencies and unnecessary functions/comments from `utils`, `tests`, `exceptions`, and `datachecks` :pr:`1`


**v0.1.1 July 3, 2023**

* *First Release*
